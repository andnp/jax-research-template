"""Sludge blanket control benchmark using Takács 10-layer settler.

The agent controls the underflow rate of a secondary clarifier to maintain
a stable sludge blanket under varying hydraulic loading. The feed TSS is
roughly constant (MLSS from an activated sludge process), but flow varies
diurnally.

Control challenge:
  - High flow → increased upward velocity → blanket rises → risk of washout
  - Low flow → blanket drops → underflow over-concentrated
  - Too high underflow → wasted pumping energy, blanket too low
  - Too low underflow → blanket rises, effluent TSS spikes

Action (1D): normalised underflow rate [Q_u / Q_u_max]
Observation (4D) — all from sensor readings:
  sensed_blanket_height / depth  (concentration analyzer)
  sensed_eff_tss / 100           (concentration analyzer)
  sensed_und_tss / 10000         (concentration analyzer)
  q_feed / q_mean
"""
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.actuators.dose_pump import DosePumpParams, DosePumpState
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.disturbances.schedule import DisturbanceSchedule, create_empty
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import step as ra_step
from process_control.units.takacs_settler import TakacsSettlerParams, TakacsSettlerState, compute_blanket_height, get_effluent_tss, get_underflow_tss
from process_control.units.takacs_settler import reset as settler_reset
from process_control.units.takacs_settler import step as settler_step


@dataclass(frozen=True)
class SludgeBlanketConfig:
    dt: float = 0.02  # hours (~1.2 min)

    # Feed: constant MLSS from activated sludge process
    feed_tss: float = 3500.0  # g/m³ (typical MLSS)

    # Flow variation
    mean_flow: float = 769.0
    diurnal_amplitude: float = 150.0
    min_flow: float = 500.0
    max_flow: float = 1050.0
    demand_noise_std: float = 1.0
    drift_scale: float = 0.05
    steps_per_day: int = 1200

    # Underflow actuator
    q_u_min: float = 100.0  # m³/h
    q_u_max: float = 1500.0  # m³/h
    q_u_ramp_rate: float = 200.0  # m³/h per hour

    # Settler geometry and settling parameters
    settler: TakacsSettlerParams = TakacsSettlerParams()

    # Blanket height threshold for observation
    blanket_threshold: float = 1500.0  # g/m³

    # Reward weights
    reward_w_eff: float = 1.0  # effluent TSS penalty
    reward_w_blanket: float = 2.0  # blanket height penalty (above 70% depth)
    reward_w_energy: float = 0.01  # pumping penalty

    # Effluent TSS limit for reward scaling
    eff_tss_limit: float = 30.0  # g/m³ (regulatory limit)

    max_disturbance_events: int = 16

    # Sensor parameters (set all to zero / sample_period=1 for ideal sensors)
    # TSS analyzers (blanket height, effluent TSS, underflow TSS)
    analyzer_noise_std: float = 5.0  # g/m³ TSS noise
    analyzer_lag: float = 0.8  # first-order lag coefficient
    analyzer_sample_period: int = 8  # steps between samples (~10 min)


@jax_dataclass
class SludgeBlanketSensorState:
    blanket_height: ResidualAnalyzerState
    eff_tss: ResidualAnalyzerState
    und_tss: ResidualAnalyzerState


@jax_dataclass
class SludgeBlanketPlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    settler_state: TakacsSettlerState
    underflow_actuator: DosePumpState
    disturbance_schedule: DisturbanceSchedule
    sensors: SludgeBlanketSensorState


def make_sludge_blanket_benchmark(
    config: SludgeBlanketConfig,
) -> tuple[
    Callable[[jax.Array], tuple[SludgeBlanketPlantState, jax.Array]],
    Callable[[SludgeBlanketPlantState, jax.Array, jax.Array], tuple[SludgeBlanketPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    settler_params = config.settler

    pump_params = DosePumpParams(
        max_dose=config.q_u_max,
        min_dose=config.q_u_min,
        max_ramp_rate=config.q_u_ramp_rate,
    )

    source_params = DiurnalSourceParams(
        mean_flow=config.mean_flow,
        diurnal_amplitude=config.diurnal_amplitude,
        min_flow=config.min_flow,
        max_flow=config.max_flow,
        demand_offset=0.0,
        flow_demand_coefficient=0.0,
        demand_noise_std=config.demand_noise_std,
        drift_scale=config.drift_scale,
        steps_per_day=config.steps_per_day,
    )

    dt = jnp.array(config.dt)
    mean_flow = jnp.array(config.mean_flow)
    feed_tss = jnp.array(config.feed_tss)
    q_u_max = jnp.array(config.q_u_max)
    depth = jnp.array(config.settler.depth)
    blanket_threshold = config.blanket_threshold
    eff_tss_limit = jnp.array(config.eff_tss_limit)

    analyzer_params = ResidualAnalyzerParams(
        noise_std=config.analyzer_noise_std,
        lag_coefficient=config.analyzer_lag,
        sample_period=config.analyzer_sample_period,
    )

    def reset(rng_key: jax.Array) -> tuple[SludgeBlanketPlantState, jax.Array]:
        k1, k2, k3 = jax.random.split(rng_key, 3)

        src = source_reset(k1)
        settler = settler_reset(config.feed_tss, settler_params, k2)
        pump = dose_pump_reset(k3)

        sensors = SludgeBlanketSensorState(
            blanket_height=ResidualAnalyzerState.create(),
            eff_tss=ResidualAnalyzerState.create(),
            und_tss=ResidualAnalyzerState.create(),
        )

        plant_state = SludgeBlanketPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            settler_state=settler,
            underflow_actuator=pump,
            disturbance_schedule=create_empty(config.max_disturbance_events),
            sensors=sensors,
        )

        blanket_h = compute_blanket_height(settler, settler_params, blanket_threshold)
        eff_tss = get_effluent_tss(settler)
        und_tss = get_underflow_tss(settler)

        obs = jnp.array(
            [
                blanket_h / depth,
                eff_tss / 100.0,
                und_tss / 10000.0,
                mean_flow / mean_flow,
            ]
        )
        return plant_state, obs

    def step(
        state: SludgeBlanketPlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[SludgeBlanketPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k1, k_sensors = jax.random.split(rng_key)
        k_bh, k_eff, k_und = jax.random.split(k_sensors, 3)

        # 1. Flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Underflow actuator (action is normalised Q_u)
        new_pump, q_u = dose_pump_step(state.underflow_actuator, action[0], pump_params, dt)

        # 3. Settler step
        new_settler = settler_step(
            state.settler_state,
            feed_tss,
            q_in,
            q_u,
            settler_params,
            dt,
        )

        # 4. True measurements
        blanket_h = compute_blanket_height(new_settler, settler_params, blanket_threshold)
        eff_tss = get_effluent_tss(new_settler)
        und_tss = get_underflow_tss(new_settler)

        # 5. Sensors
        new_bh_sensor, bh_reading = ra_step(state.sensors.blanket_height, blanket_h, analyzer_params, k_bh)
        new_eff_sensor, eff_reading = ra_step(state.sensors.eff_tss, eff_tss, analyzer_params, k_eff)
        new_und_sensor, und_reading = ra_step(state.sensors.und_tss, und_tss, analyzer_params, k_und)

        new_sensors = SludgeBlanketSensorState(
            blanket_height=new_bh_sensor,
            eff_tss=new_eff_sensor,
            und_tss=new_und_sensor,
        )

        # 6. Observation (from sensor readings)
        obs = jnp.array(
            [
                bh_reading / depth,
                eff_reading / 100.0,
                und_reading / 10000.0,
                q_in / mean_flow,
            ]
        )

        # 7. Reward (from sensor readings)
        eff_penalty = (eff_reading / eff_tss_limit) ** 2

        blanket_frac = bh_reading / depth
        blanket_penalty = jnp.maximum(0.0, blanket_frac - 0.7) ** 2

        energy_penalty = q_u / q_u_max

        reward = -(config.reward_w_eff * eff_penalty + config.reward_w_blanket * blanket_penalty + config.reward_w_energy * energy_penalty)

        new_state = SludgeBlanketPlantState(
            step_count=state.step_count + 1,
            source_state=new_source,
            settler_state=new_settler,
            underflow_actuator=new_pump,
            disturbance_schedule=state.disturbance_schedule,
            sensors=new_sensors,
        )

        info: dict[str, jax.Array] = {
            "effluent_tss": eff_tss,
            "underflow_tss": und_tss,
            "blanket_height": blanket_h,
            "q_underflow": q_u,
            "q_in": q_in,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
