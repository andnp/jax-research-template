"""Biological phosphorus removal benchmark (ASM2d-based).

A simplified activated sludge plant with anaerobic/anoxic/aerobic zones
for enhanced biological phosphorus removal (EBPR). PAOs store PHA
anaerobically (releasing P) and take up P aerobically.

Action (2D): [Q_recycle / Q_max, carbon_dose / dose_max]
  - Q_recycle: internal recycle from aerobic → anaerobic zone
  - carbon_dose: external carbon (VFA) to anaerobic zone (helps PAOs)

Observation (7D):
  effluent PO₄ / 10
  effluent NH₄ / 35
  anaerobic PO₄ / 30  (high = good PAO activity)
  aerobic DO / 8
  aerobic PHA/PAO ratio
  Q_recycle / Q_max
  carbon_dose / dose_max
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.asm2d import ASM2dParams, reactions_asm2d


@jax_dataclass
class BioPBenchState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    anaerobic_state: jax.Array  # (N_COMPONENTS,) anaerobic zone
    anoxic_state: jax.Array  # (N_COMPONENTS,) anoxic zone
    aerobic_state: jax.Array  # (N_COMPONENTS,) aerobic zone
    po4_eff_sensor: ResidualAnalyzerState
    nh4_eff_sensor: ResidualAnalyzerState
    po4_ana_sensor: ResidualAnalyzerState
    last_q_in: jax.Array


@dataclass(frozen=True)
class BioPConfig:
    dt: float = 0.02  # hours
    steps_per_day: int = 1200

    # Reactor volumes
    v_anaerobic: float = 500.0  # m³
    v_anoxic: float = 1000.0  # m³
    v_aerobic: float = 2000.0  # m³

    # Aeration in aerobic zone
    kla_aerobic: float = 6.0  # h⁻¹ (fixed, not action-controlled for simplicity)
    s_o_sat: float = 8.0  # mg O₂/L

    # Flow
    mean_flow: float = 500.0  # m³/h
    diurnal_amplitude: float = 100.0
    min_flow: float = 300.0
    max_flow: float = 700.0
    drift_scale: float = 0.1

    # Recycle
    q_recycle_max: float = 1500.0  # m³/h (3× Q_in)

    # External carbon
    carbon_dose_max: float = 50.0  # mg COD/L

    # ASM2d parameters
    asm2d: ASM2dParams = ASM2dParams()

    # Sensors
    po4_noise_std: float = 0.1
    po4_lag: float = 0.7
    nh4_noise_std: float = 0.3
    nh4_lag: float = 0.7
    sensor_sample_period: int = 5

    # Reward
    reward_w_po4: float = 1.0
    reward_w_nh4: float = 0.5
    reward_w_carbon: float = 0.05
    po4_limit: float = 1.0  # mg P/L effluent limit
    nh4_limit: float = 4.0  # mg N/L effluent limit


def _mix_asm2d(state_a: jax.Array, q_a: jax.Array, state_b: jax.Array, q_b: jax.Array):
    """Mix two ASM2d state vectors by flow-weighted average."""
    q_total = q_a + q_b + 1e-10
    return (state_a * q_a + state_b * q_b) / q_total


def make_bio_p_benchmark(config: BioPConfig):
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
        lag_coefficient=config.po4_lag,
        sample_period=config.sensor_sample_period,
    )
    nh4_analyzer = ResidualAnalyzerParams(
        noise_std=config.nh4_noise_std,
        lag_coefficient=config.nh4_lag,
        sample_period=config.sensor_sample_period,
    )

    q_max = jnp.array(config.q_recycle_max)
    carbon_max = jnp.array(config.carbon_dose_max)

    # Default influent (ASM2d: 17 components)
    from process_control.units.asm2d import make_default_influent_asm2d

    influent = make_default_influent_asm2d()

    def _reactor_step(state: jax.Array, inlet: jax.Array, q_total: jax.Array, volume: float, kla: jax.Array):
        """One CSTR step with ASM2d reactions."""
        dilution = q_total / volume
        dc_reactions = reactions_asm2d(state, config.asm2d)
        dc_dilution = dilution * (inlet - state)

        # Aeration for DO
        s_o = state[7]
        dc_aeration = kla * (config.s_o_sat - s_o)

        new_state = state + (dc_reactions + dc_dilution) * config.dt
        new_state = new_state.at[7].add(dc_aeration * config.dt)
        return jnp.maximum(new_state, 0.0)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)
        src = source_reset(k1)

        # Initial states: start with influent-like concentrations + biomass
        init_state = influent.at[4].set(2500.0)  # X_BH
        init_state = init_state.at[5].set(150.0)  # X_BA
        init_state = init_state.at[14].set(500.0)  # X_PAO
        init_state = init_state.at[15].set(100.0)  # X_PHA
        init_state = init_state.at[16].set(100.0)  # X_PP

        state = BioPBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            anaerobic_state=init_state.at[7].set(0.0),  # no DO in anaerobic
            anoxic_state=init_state.at[7].set(0.0),
            aerobic_state=init_state.at[7].set(2.0),  # DO ~2 in aerobic
            po4_eff_sensor=analyzer_reset(k2),
            nh4_eff_sensor=analyzer_reset(k3),
            po4_ana_sensor=analyzer_reset(k4),
            last_q_in=jnp.array(config.mean_flow),
        )
        obs = jnp.array(
            [
                6.0 / 10.0,  # initial effluent PO₄
                7.0 / 35.0,  # initial effluent NH₄
                15.0 / 30.0,  # anaerobic PO₄
                2.0 / 8.0,  # aerobic DO
                0.2,  # PHA/PAO ratio
                0.5,  # Q_recycle / Q_max
                0.0,  # carbon dose
            ]
        )
        return state, obs

    def step(state: BioPBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Actions
        q_recycle = jnp.clip(action[0], 0.0, 1.0) * q_max
        carbon_dose = jnp.clip(action[1], 0.0, 1.0) * carbon_max

        # 3. Anaerobic zone: influent + recycle from aerobic (no aeration)
        inf_with_carbon = influent.at[1].set(influent[1] + carbon_dose)  # add VFA to S_S
        anaerobic_inlet = _mix_asm2d(inf_with_carbon, q_in, state.aerobic_state, q_recycle)
        new_anaerobic = _reactor_step(
            state.anaerobic_state,
            anaerobic_inlet,
            q_in + q_recycle,
            config.v_anaerobic,
            jnp.array(0.0),
        )

        # 4. Anoxic zone: from anaerobic (no aeration)
        new_anoxic = _reactor_step(
            state.anoxic_state,
            new_anaerobic,
            q_in + q_recycle,
            config.v_anoxic,
            jnp.array(0.0),
        )

        # 5. Aerobic zone: from anoxic (with aeration)
        new_aerobic = _reactor_step(
            state.aerobic_state,
            new_anoxic,
            q_in + q_recycle,
            config.v_aerobic,
            jnp.array(config.kla_aerobic),
        )

        # 6. Effluent (simplified: no settler, just aerobic effluent soluble fractions)
        eff_po4 = new_aerobic[13]  # S_PO4
        eff_nh4 = new_aerobic[9]  # S_NH
        ana_po4 = new_anaerobic[13]
        aerobic_do = new_aerobic[7]
        pha_pao_ratio = new_aerobic[15] / (new_aerobic[14] + 1e-10)

        # 7. Sensors
        new_po4_eff, sensed_po4_eff = analyzer_step(state.po4_eff_sensor, eff_po4, po4_analyzer, k2)
        new_nh4_eff, sensed_nh4_eff = analyzer_step(state.nh4_eff_sensor, eff_nh4, nh4_analyzer, k3)
        new_po4_ana, sensed_po4_ana = analyzer_step(state.po4_ana_sensor, ana_po4, po4_analyzer, k4)

        # 8. Observation
        obs = jnp.array(
            [
                sensed_po4_eff / 10.0,
                sensed_nh4_eff / 35.0,
                sensed_po4_ana / 30.0,
                aerobic_do / 8.0,
                pha_pao_ratio,
                q_recycle / q_max,
                carbon_dose / carbon_max,
            ]
        )

        # 9. Reward
        po4_violation = jnp.maximum(sensed_po4_eff - config.po4_limit, 0.0)
        nh4_violation = jnp.maximum(sensed_nh4_eff - config.nh4_limit, 0.0)
        reward = -(config.reward_w_po4 * po4_violation**2 + config.reward_w_nh4 * nh4_violation**2 + config.reward_w_carbon * (carbon_dose / carbon_max))

        new_state = BioPBenchState(
            step_count=state.step_count + 1,
            source_state=new_source,
            anaerobic_state=new_anaerobic,
            anoxic_state=new_anoxic,
            aerobic_state=new_aerobic,
            po4_eff_sensor=new_po4_eff,
            nh4_eff_sensor=new_nh4_eff,
            po4_ana_sensor=new_po4_ana,
            last_q_in=q_in,
        )
        info: dict[str, jax.Array] = {
            "po4_eff": eff_po4,
            "nh4_eff": eff_nh4,
            "po4_anaerobic": ana_po4,
            "do_aerobic": aerobic_do,
            "x_pao": new_aerobic[14],
            "x_pha": new_aerobic[15],
            "x_pp": new_aerobic[16],
            "q_in": q_in,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
