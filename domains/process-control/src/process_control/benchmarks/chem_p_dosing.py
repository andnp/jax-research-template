"""Chemical phosphorus removal benchmark.

Single-stage FeCl₃ dosing for PO₄-P removal with a clear cost–quality
trade-off.  The agent controls the iron dose (1D action) to minimise
effluent phosphate while avoiding excessive chemical usage.

Plant model:
  1. Diurnal influent (flow + PO₄-P via SinusoidalChannel)
  2. FeCl₃ dosing via DosingSystem (PO₄ sensor → PI → pump)
  3. Instantaneous precipitation (Monod-type saturation)
  4. Observation: effluent PO₄, influent PO₄, flow, dose, flow Δ
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.dosing_system import (
    DIRECT,
    DosingSystemParams,
    DosingSystemState,
)
from process_control.actuators.dosing_system import reset as dosing_reset
from process_control.actuators.dosing_system import step as dosing_step
from process_control.chemistry.precipitation import PrecipitationParams, precipitate
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step

# ── State ──────────────────────────────────────────────────────────


@jax_dataclass
class ChemPDosingState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    po4_drift: jax.Array  # drift for influent PO₄ channel
    dosing_loop: DosingSystemState
    po4_inf_sensor: ResidualAnalyzerState
    last_po4_eff: jax.Array  # true effluent PO₄ (for sensor input)
    last_q_in: jax.Array


# ── Config ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChemPDosingConfig:
    dt: float = 0.02  # hours per step (≈ 1 min)
    target_po4: float = 0.5  # mg-P/L effluent target

    control_mode: int = DIRECT

    # ── Influent PO₄-P channel ────────────────────────────────────
    po4_mean: float = 5.0  # mg-P/L mean influent PO₄
    po4_amplitude: float = 2.0  # diurnal swing
    po4_min: float = 1.0
    po4_max: float = 10.0
    po4_drift_scale: float = 0.2

    # ── FeCl₃ dosing pump ─────────────────────────────────────────
    dose_max: float = 30.0  # mg-Fe/L maximum dose
    dose_min: float = 0.0
    dose_ramp_rate: float = 15.0  # mg-Fe/L/h symmetric for chemical pump

    # ── Dosing loop PI ────────────────────────────────────────────
    pi_kp: float = 2.0
    pi_ki: float = 0.5
    pi_ff: float = 10.0  # feedforward: typical dose for mean PO₄
    pi_output_min: float = 0.0
    pi_output_max: float = 30.0
    pi_max_integral: float = 20.0
    pi_base_setpoint: float = 0.5

    # ── Dosing loop PO₄ sensor (effluent) ─────────────────────────
    eff_sensor_noise_std: float = 0.05
    eff_sensor_lag: float = 0.7
    eff_sensor_drift_rate: float = 0.002

    # ── Standalone influent PO₄ analyzer ──────────────────────────
    inf_sensor_noise_std: float = 0.1
    inf_sensor_lag: float = 0.5
    inf_sensor_sample_period: int = 5

    # ── Precipitation chemistry ───────────────────────────────────
    stoich_fe_per_p: float = 1.8
    k_half: float = 0.5
    precip_efficiency: float = 0.85
    p_min: float = 0.01

    # ── Flow source ───────────────────────────────────────────────
    mean_flow: float = 75.0
    diurnal_amplitude: float = 20.0
    min_flow: float = 50.0
    max_flow: float = 100.0
    drift_scale: float = 0.2
    steps_per_day: int = 1200  # 0.02h steps × 1200 = 24h

    # ── Reward ────────────────────────────────────────────────────
    reward_w_po4: float = 1.0  # weight on (po4_eff - target)²
    reward_w_cost: float = 0.002  # weight on normalised Fe cost


def make_chem_p_dosing_benchmark(
    config: ChemPDosingConfig,
):
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

    po4_channel_params = ChannelParams(
        mean=config.po4_mean,
        amplitude=config.po4_amplitude,
        min_value=config.po4_min,
        max_value=config.po4_max,
        drift_scale=config.po4_drift_scale,
        drift_clip=config.po4_amplitude,
    )

    dosing_params = DosingSystemParams(
        control_mode=config.control_mode,
        base_setpoint=config.pi_base_setpoint,
        sensor_noise_std=config.eff_sensor_noise_std,
        sensor_lag=config.eff_sensor_lag,
        sensor_drift_rate=config.eff_sensor_drift_rate,
        kp=config.pi_kp,
        ki=config.pi_ki,
        ff=config.pi_ff,
        output_min=config.pi_output_min,
        output_max=config.pi_output_max,
        max_integral=config.pi_max_integral,
        max_ramp_up=config.dose_ramp_rate,
        max_ramp_down=config.dose_ramp_rate,
    )

    precip_params = PrecipitationParams(
        stoich_fe_per_p=config.stoich_fe_per_p,
        k_half=config.k_half,
        efficiency=config.precip_efficiency,
        p_min=config.p_min,
    )

    inf_analyzer_params = ResidualAnalyzerParams(
        noise_std=config.inf_sensor_noise_std,
        lag_coefficient=config.inf_sensor_lag,
        sample_period=config.inf_sensor_sample_period,
    )

    dt = jnp.array(config.dt)
    q_mean = jnp.array(config.mean_flow)
    dose_max = jnp.array(config.dose_max)
    target_po4 = jnp.array(config.target_po4)
    po4_norm = jnp.array(config.po4_max)
    w_po4 = jnp.array(config.reward_w_po4)
    w_cost = jnp.array(config.reward_w_cost)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        src_state = source_reset(k1)
        ds_state = dosing_reset(config.po4_mean, config.pi_ff, k2)
        inf_sensor = analyzer_reset(k3)

        state = ChemPDosingState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src_state,
            po4_drift=jnp.array(0.0),
            dosing_loop=ds_state,
            po4_inf_sensor=inf_sensor,
            last_po4_eff=jnp.array(config.po4_mean),
            last_q_in=jnp.array(config.mean_flow),
        )

        obs = jnp.array(
            [
                config.po4_mean / config.po4_max,  # effluent PO₄ (initial)
                1.0,  # q_in / q_mean
                0.0,  # fe_dose / dose_max
                config.po4_mean / config.po4_max,  # influent PO₄
                0.0,  # dq/dt
            ]
        )
        return state, obs

    def step(
        state: ChemPDosingState,
        action: jax.Array,
        rng_key: jax.Array,
    ):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Source generates flow
        new_source, _transport, flow, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Influent PO₄ via sinusoidal channel (phase-correlated with flow)
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        po4_signal = jnp.sin(2.0 * jnp.pi * t)

        new_po4_drift, po4_inf = channel_step(
            state.po4_drift,
            po4_signal,
            po4_channel_params,
            k2,
        )

        # 3. Dosing loop reads PREVIOUS effluent PO₄
        new_dosing, sensed_eff_po4, fe_dose, pi_dose = dosing_step(
            state.dosing_loop,
            action[0],
            state.last_po4_eff,
            dosing_params,
            dt,
            k3,
        )

        # 4. Precipitation: influent PO₄ + FeCl₃ → effluent PO₄
        po4_eff, fe_consumed = precipitate(po4_inf, fe_dose, precip_params)

        # 5. Influent PO₄ analyzer (standalone sensor)
        new_inf_sensor, sensed_inf_po4 = analyzer_step(
            state.po4_inf_sensor,
            po4_inf,
            inf_analyzer_params,
            k4,
        )

        # 6. Observation (5D)
        dq = (flow - state.last_q_in) / q_mean
        obs = jnp.array(
            [
                sensed_eff_po4 / po4_norm,
                flow / q_mean,
                fe_dose / dose_max,
                sensed_inf_po4 / po4_norm,
                dq,
            ]
        )

        # 7. Reward: penalise effluent violation + chemical cost
        violation = jnp.maximum(sensed_eff_po4 - target_po4, 0.0)
        reward = -(w_po4 * violation**2 + w_cost * (fe_dose / dose_max))

        new_state = ChemPDosingState(
            step_count=state.step_count + 1,
            source_state=new_source,
            po4_drift=new_po4_drift,
            dosing_loop=new_dosing,
            po4_inf_sensor=new_inf_sensor,
            last_po4_eff=po4_eff,
            last_q_in=flow,
        )

        info: dict[str, jax.Array] = {
            "po4_eff": po4_eff,
            "po4_inf": po4_inf,
            "fe_dose": fe_dose,
            "fe_consumed": fe_consumed,
            "flow": flow,
        }

        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
