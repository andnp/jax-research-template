"""Combined nitrogen and phosphorus control benchmark (ASM2d + FeCl₃).

The hardest variant: biological N removal (nitrification/denitrification) and
biological + chemical P removal in a single plant. Conflicting requirements:
  - High aeration helps nitrification but hurts anaerobic PAO zone
  - Denitrification needs anoxic conditions but NO₃ intrusion kills bio-P
  - Chemical P backup (FeCl₃) can compensate but costs money

Action (4D): [kla / kla_max, Q_recycle / Q_max, FeCl₃ dose / dose_max, carbon_dose / carbon_max]
Observation (9D):
  effluent PO₄ / 10
  effluent NH₄ / 35
  effluent NO₃ / 20
  anaerobic PO₄ / 30
  aerobic DO / 8
  PHA/PAO ratio
  flow / mean_flow
  FeCl₃ dose / dose_max
  carbon dose / carbon_max
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.precipitation import PrecipitationParams, precipitate
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.asm2d import ASM2dParams, make_default_influent_asm2d, reactions_asm2d


@jax_dataclass
class CombinedNPState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    anaerobic_state: jax.Array
    anoxic_state: jax.Array
    aerobic_state: jax.Array
    po4_eff_sensor: ResidualAnalyzerState
    nh4_eff_sensor: ResidualAnalyzerState
    no3_eff_sensor: ResidualAnalyzerState
    po4_ana_sensor: ResidualAnalyzerState
    last_q_in: jax.Array


@dataclass(frozen=True)
class CombinedNPConfig:
    dt: float = 0.02
    steps_per_day: int = 1200

    # Reactor volumes
    v_anaerobic: float = 500.0
    v_anoxic: float = 1000.0
    v_aerobic: float = 2000.0

    # Aeration range
    kla_max: float = 10.0
    s_o_sat: float = 8.0

    # Flow
    mean_flow: float = 500.0
    diurnal_amplitude: float = 100.0
    min_flow: float = 300.0
    max_flow: float = 700.0
    drift_scale: float = 0.1

    # Recycle
    q_recycle_max: float = 1500.0

    # Chemical P (FeCl₃)
    fe_dose_max: float = 30.0
    precip: PrecipitationParams = PrecipitationParams()

    # External carbon
    carbon_dose_max: float = 50.0

    # ASM2d
    asm2d: ASM2dParams = ASM2dParams()

    # Sensors
    po4_noise_std: float = 0.1
    nh4_noise_std: float = 0.3
    no3_noise_std: float = 0.3
    sensor_lag: float = 0.7
    sensor_sample_period: int = 5

    # Reward
    reward_w_po4: float = 1.0
    reward_w_nh4: float = 1.0
    reward_w_no3: float = 0.3
    reward_w_fe_cost: float = 0.05
    reward_w_carbon_cost: float = 0.03
    reward_w_aeration: float = 0.02
    po4_limit: float = 1.0
    nh4_limit: float = 4.0
    no3_limit: float = 10.0


def _mix_asm2d(state_a: jax.Array, q_a: jax.Array, state_b: jax.Array, q_b: jax.Array):
    q_total = q_a + q_b + 1e-10
    return (state_a * q_a + state_b * q_b) / q_total


def make_combined_np_benchmark(config: CombinedNPConfig):
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
    po4_analyzer = ResidualAnalyzerParams(
        noise_std=config.po4_noise_std,
        lag_coefficient=config.sensor_lag,
        sample_period=config.sensor_sample_period,
    )
    nh4_analyzer = ResidualAnalyzerParams(
        noise_std=config.nh4_noise_std,
        lag_coefficient=config.sensor_lag,
        sample_period=config.sensor_sample_period,
    )
    no3_analyzer = ResidualAnalyzerParams(
        noise_std=config.no3_noise_std,
        lag_coefficient=config.sensor_lag,
        sample_period=config.sensor_sample_period,
    )

    q_max = jnp.array(config.q_recycle_max)
    kla_max = jnp.array(config.kla_max)
    fe_max = jnp.array(config.fe_dose_max)
    carbon_max = jnp.array(config.carbon_dose_max)
    mean_flow = jnp.array(config.mean_flow)
    influent = make_default_influent_asm2d()

    def _reactor_step(state: jax.Array, inlet: jax.Array, q_total: jax.Array, volume: float, kla: jax.Array):
        dilution = q_total / volume
        dc_reactions = reactions_asm2d(state, config.asm2d)
        dc_dilution = dilution * (inlet - state)
        s_o = state[7]
        dc_aeration = kla * (config.s_o_sat - s_o)
        new_state = state + (dc_reactions + dc_dilution) * config.dt
        new_state = new_state.at[7].add(dc_aeration * config.dt)
        return jnp.maximum(new_state, 0.0)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)
        src = source_reset(k1)

        init_state = influent.at[4].set(2500.0).at[5].set(150.0)
        init_state = init_state.at[14].set(500.0).at[15].set(100.0).at[16].set(100.0)

        state = CombinedNPState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            anaerobic_state=init_state.at[7].set(0.0),
            anoxic_state=init_state.at[7].set(0.0),
            aerobic_state=init_state.at[7].set(2.0),
            po4_eff_sensor=analyzer_reset(k2),
            nh4_eff_sensor=analyzer_reset(k3),
            no3_eff_sensor=analyzer_reset(k4),
            po4_ana_sensor=analyzer_reset(k5),
            last_q_in=jnp.array(config.mean_flow),
        )
        obs = jnp.zeros(9)
        return state, obs

    def step(state: CombinedNPState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

        # 1. Flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Actions
        kla = jnp.clip(action[0], 0.0, 1.0) * kla_max
        q_recycle = jnp.clip(action[1], 0.0, 1.0) * q_max
        fe_dose = jnp.clip(action[2], 0.0, 1.0) * fe_max
        carbon_dose = jnp.clip(action[3], 0.0, 1.0) * carbon_max

        # 3. Anaerobic zone: influent + recycle, add external carbon
        inf_with_carbon = influent.at[1].set(influent[1] + carbon_dose)
        anaerobic_inlet = _mix_asm2d(inf_with_carbon, q_in, state.aerobic_state, q_recycle)
        new_anaerobic = _reactor_step(
            state.anaerobic_state,
            anaerobic_inlet,
            q_in + q_recycle,
            config.v_anaerobic,
            jnp.array(0.0),
        )

        # 4. Anoxic zone
        new_anoxic = _reactor_step(
            state.anoxic_state,
            new_anaerobic,
            q_in + q_recycle,
            config.v_anoxic,
            jnp.array(0.0),
        )

        # 5. Aerobic zone (with variable aeration)
        new_aerobic = _reactor_step(
            state.aerobic_state,
            new_anoxic,
            q_in + q_recycle,
            config.v_aerobic,
            kla,
        )

        # 6. Chemical P removal on effluent PO₄
        bio_po4_eff = new_aerobic[13]
        final_po4, fe_consumed = precipitate(bio_po4_eff, fe_dose, config.precip)

        # 7. Effluent quality
        eff_nh4 = new_aerobic[9]
        eff_no3 = new_aerobic[8]
        ana_po4 = new_anaerobic[13]
        aerobic_do = new_aerobic[7]
        pha_pao_ratio = new_aerobic[15] / (new_aerobic[14] + 1e-10)

        # 8. Sensors
        new_po4_s, sensed_po4 = analyzer_step(state.po4_eff_sensor, final_po4, po4_analyzer, k2)
        new_nh4_s, sensed_nh4 = analyzer_step(state.nh4_eff_sensor, eff_nh4, nh4_analyzer, k3)
        new_no3_s, sensed_no3 = analyzer_step(state.no3_eff_sensor, eff_no3, no3_analyzer, k4)
        new_po4a_s, sensed_ana_po4 = analyzer_step(state.po4_ana_sensor, ana_po4, po4_analyzer, k5)

        # 9. Observation
        obs = jnp.array(
            [
                sensed_po4 / 10.0,
                sensed_nh4 / 35.0,
                sensed_no3 / 20.0,
                sensed_ana_po4 / 30.0,
                aerobic_do / 8.0,
                pha_pao_ratio,
                q_in / mean_flow,
                fe_dose / fe_max,
                carbon_dose / carbon_max,
            ]
        )

        # 10. Reward
        po4_v = jnp.maximum(sensed_po4 - config.po4_limit, 0.0)
        nh4_v = jnp.maximum(sensed_nh4 - config.nh4_limit, 0.0)
        no3_v = jnp.maximum(sensed_no3 - config.no3_limit, 0.0)
        reward = -(
            config.reward_w_po4 * po4_v**2
            + config.reward_w_nh4 * nh4_v**2
            + config.reward_w_no3 * no3_v**2
            + config.reward_w_fe_cost * (fe_dose / fe_max)
            + config.reward_w_carbon_cost * (carbon_dose / carbon_max)
            + config.reward_w_aeration * (kla / kla_max)
        )

        new_state = CombinedNPState(
            step_count=state.step_count + 1,
            source_state=new_source,
            anaerobic_state=new_anaerobic,
            anoxic_state=new_anoxic,
            aerobic_state=new_aerobic,
            po4_eff_sensor=new_po4_s,
            nh4_eff_sensor=new_nh4_s,
            no3_eff_sensor=new_no3_s,
            po4_ana_sensor=new_po4a_s,
            last_q_in=q_in,
        )
        info: dict[str, jax.Array] = {
            "po4_eff": final_po4,
            "nh4_eff": eff_nh4,
            "no3_eff": eff_no3,
            "po4_anaerobic": ana_po4,
            "do_aerobic": aerobic_do,
            "x_pao": new_aerobic[14],
            "x_pha": new_aerobic[15],
            "x_pp": new_aerobic[16],
            "fe_consumed": fe_consumed,
            "q_in": q_in,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
