"""Reject water management benchmark.

Controls the timing and rate of high-ammonia return flow from sludge
dewatering back to the headworks of the main treatment plant. Poorly
timed reject water return causes NH₄ spikes that overload nitrification.

The agent must learn to spread reject water return over low-load periods.

Action (1D): [Q_reject / Q_max]
  - Q_reject controls the instantaneous return rate.

Observation (6D):
  main_plant_nh4 / 10
  tank_level / reject_tank_max
  time_of_day (0-1)
  main_flow / mean_flow
  reject_flow / q_reject_max
  dQ_main/dt
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.lagged import LaggedSensorParams, LaggedSensorState
from process_control.sensors.lagged import reset as analyzer_reset
from process_control.sensors.lagged import step as analyzer_step


@jax_dataclass
class RejectWaterBenchState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    main_nh4: jax.Array  # main plant NH₄ (simplified model)
    reject_tank_volume: jax.Array  # reject water stored (m³)
    main_nh4_sensor: LaggedSensorState
    last_q_main: jax.Array


@dataclass(frozen=True)
class RejectWaterConfig:
    dt: float = 0.02
    steps_per_day: int = 1200

    # Main plant flow. Assumed benchmark regime, not a sourced series: a wide
    # swing makes return timing matter at peak load.
    mean_flow: float = 769.0
    diurnal_amplitude: float = 400.0
    min_flow: float = 400.0
    max_flow: float = 1200.0
    drift_scale: float = 0.05

    # Reject water characteristics. Digester centrate ammonia is typically
    # 500-1500 mg-N/L (Fux & Siegrist 2004, Water Sci. Technol. 50(10):19-26;
    # https://pubmed.ncbi.nlm.nih.gov/15656291/); 400 was below that range.
    # Nitrifying it also consumes alkalinity (7.14 g CaCO3/g N oxidised,
    # ASM1 stoichiometry) — not tracked by this reduced-order model.
    reject_nh4: float = 800.0  # mg-N/L
    reject_production: float = 8.0  # m³/h continuous production from dewatering
    reject_tank_max: float = 200.0  # m³ buffer tank
    q_reject_max: float = 30.0  # m³/h max return rate

    # Reduced CSTR for the nitrifying zone. Volume is the BSM1 aerobic volume
    # (three 1333 m³ reactors; Alex et al. 2008, BSM1 technical report).
    plant_volume: float = 4000.0  # m³
    influent_nh4: float = 31.56  # mg-N/L (BSM1 dry-weather mean)
    # Assumed, calibrated for this benchmark (not a BSM1 kinetic rate). Chosen
    # so the full balance, dilution included, settles near 1.6 mg-N/L with no
    # return and near 3 mg-N/L at mean flow with pass-through return.
    nitrification_rate: float = 9.5  # mg-N/L/h max rate (r_max)
    k_nh: float = 1.0  # mg-N/L half-saturation (ASM1 K_NH, Henze et al. 1987)
    nh4_baseline: float = 2.0  # mg-N/L initial main-plant NH₄

    # Sensors
    nh4_noise_std: float = 0.5
    nh4_lag: float = 0.7
    nh4_sample_period: int = 5

    # Reward
    reward_w_nh4: float = 1.0  # penalty for main plant NH₄ spike
    reward_w_overflow: float = 5.0  # penalty for reject tank overflow
    # Penalises only overflow *risk* (level above this fraction of capacity),
    # not level itself — the tank exists to buffer return timing, so
    # penalising level directly pushed immediate draining instead of holding.
    reward_w_timing: float = 0.1  # penalty for tank level approaching capacity
    tank_risk_threshold: float = 0.8  # fraction of capacity where risk penalty starts
    nh4_limit: float = 4.0  # mg-N/L effluent limit (BSM1 S_NH,e = 4 g N/m³)


def make_reject_water_benchmark(config: RejectWaterConfig):
    source_params = DiurnalSourceParams(
        mean_flow=config.mean_flow,
        diurnal_amplitude=config.diurnal_amplitude,
        min_flow=config.min_flow,
        max_flow=config.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,
        demand_noise_std=0.0,
        drift_scale=config.drift_scale,
        steps_per_day=config.steps_per_day,
    )
    nh4_analyzer = LaggedSensorParams(
        noise_std=config.nh4_noise_std,
        lag_coefficient=config.nh4_lag,
        sample_period=config.nh4_sample_period,
    )

    mean_flow = jnp.array(config.mean_flow)
    q_reject_max = jnp.array(config.q_reject_max)
    tank_max = jnp.array(config.reject_tank_max)
    nh4_limit = jnp.array(config.nh4_limit)

    def reset(rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)
        src = source_reset(k1)
        initial_main_nh4 = jnp.array(config.nh4_baseline)
        state = RejectWaterBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            main_nh4=initial_main_nh4,
            reject_tank_volume=jnp.array(config.reject_tank_max * 0.5),
            main_nh4_sensor=analyzer_reset(initial_main_nh4, k2),
            last_q_main=jnp.array(config.mean_flow),
        )
        obs = jnp.array(
            [
                config.nh4_baseline / 10.0,
                0.5,  # tank level: the reset is half full
                0.0,  # time of day
                1.0,  # flow ratio
                0.0,  # reject flow
                0.0,  # dQ/dt
            ]
        )
        return state, obs

    def step(state: RejectWaterBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)

        # 1. Main plant flow
        new_source, _transport, q_main, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Reject water return rate
        q_reject = jnp.clip(action[0], 0.0, 1.0) * q_reject_max
        q_reject = jnp.minimum(q_reject, state.reject_tank_volume / config.dt)

        # 3. Reject tank mass balance
        new_tank = state.reject_tank_volume + config.reject_production * config.dt - q_reject * config.dt
        overflow = jnp.maximum(new_tank - tank_max, 0.0)
        new_tank = jnp.clip(new_tank, 0.0, tank_max)

        # 4. Main plant NH₄: CSTR balance with Monod nitrification
        dilution = (
            q_main * (config.influent_nh4 - state.main_nh4) + q_reject * (config.reject_nh4 - state.main_nh4)
        ) / config.plant_volume
        nitrification_sink = config.nitrification_rate * state.main_nh4 / (config.k_nh + state.main_nh4)
        new_main_nh4 = state.main_nh4 + (dilution - nitrification_sink) * config.dt
        new_main_nh4 = jnp.maximum(new_main_nh4, 0.0)

        # 5. Sensors
        new_main_sensor, sensed_main_nh4 = analyzer_step(
            state.main_nh4_sensor,
            new_main_nh4,
            nh4_analyzer,
            k2,
        )

        # 6. Time of day
        time_of_day = (state.step_count % config.steps_per_day) / config.steps_per_day

        # 7. Observation
        dq = (q_main - state.last_q_main) / mean_flow
        obs = jnp.array(
            [
                sensed_main_nh4 / 10.0,
                new_tank / tank_max,
                time_of_day,
                q_main / mean_flow,
                q_reject / q_reject_max,
                dq,
            ]
        )

        # 8. Reward
        nh4_violation = jnp.maximum(sensed_main_nh4 - nh4_limit, 0.0)
        reward = -(
            config.reward_w_nh4 * (nh4_violation / nh4_limit) ** 2
            + config.reward_w_overflow * (overflow / tank_max)
            + config.reward_w_timing * jnp.maximum(new_tank / tank_max - config.tank_risk_threshold, 0.0) ** 2
        )

        new_state = RejectWaterBenchState(
            step_count=state.step_count + 1,
            source_state=new_source,
            main_nh4=new_main_nh4,
            reject_tank_volume=new_tank,
            main_nh4_sensor=new_main_sensor,
            last_q_main=q_main,
        )
        info: dict[str, jax.Array] = {
            "main_nh4": new_main_nh4,
            "q_reject": q_reject,
            "tank_volume": new_tank,
            "overflow": overflow,
            "q_main": q_main,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
