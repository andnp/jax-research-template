"""Drinking water treatment train benchmark.

Multi-stage cascade: coagulation → membrane filtration → chlorine disinfection.
Upstream decisions affect downstream performance:
  - Underdosing coagulant → higher particle load on membrane → faster fouling
  - Underdosing coagulant → higher organic load → more chlorine demand
  - Overdosing coagulant → wasted chemical, possible re-stabilisation
  - Backwash timing affects throughput and downstream chlorine demand

Action (3D): [coag_dose / dose_max, backwash_trigger (>0.5 = wash), Cl2_dose / Cl2_max]
Observation (8D):
  raw_turbidity / 50
  post_coag_tss / 100
  TMP / max_TMP
  permeate_flow / max_flow
  Cl2_residual / 5
  hours_since_backwash
  coag_dose / dose_max
  flow / mean_flow
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.coagulation import CoagulationParams, coagulate
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.membrane import MembraneParams, MembraneState
from process_control.units.membrane import reset as membrane_reset
from process_control.units.membrane import step as membrane_step


@jax_dataclass
class DrinkingWaterTrainState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    turbidity_drift: jax.Array
    membrane_state: MembraneState
    cl2_residual: jax.Array  # chlorine residual at basin outlet (mg/L)
    tmp_sensor: ResidualAnalyzerState
    cl2_sensor: ResidualAnalyzerState
    last_q_in: jax.Array


@dataclass(frozen=True)
class DrinkingWaterTrainConfig:
    dt: float = 0.01
    steps_per_day: int = 2400

    # Raw water source
    mean_flow: float = 200.0  # m³/h
    diurnal_amplitude: float = 40.0
    min_flow: float = 120.0
    max_flow: float = 300.0
    drift_scale: float = 0.2

    # Raw water turbidity
    turb_mean: float = 10.0  # NTU
    turb_amplitude: float = 5.0
    turb_min: float = 2.0
    turb_max: float = 50.0  # storm events
    turb_drift_scale: float = 0.4

    # Coagulation
    coag: CoagulationParams = CoagulationParams()
    coag_dose_max: float = 80.0  # mg/L

    # Membrane
    membrane: MembraneParams = MembraneParams(
        area=50.0,  # smaller plant
        r_membrane=1e11,
        k_rev_fouling=3e8,  # lower fouling rate (coag-treated water)
        k_irr_fouling=5e5,
    )
    flux_max: float = 0.06  # m/h

    # Chlorine disinfection (simplified contact basin)
    cl2_dose_max: float = 5.0  # mg/L
    cl2_decay_rate: float = 0.3  # 1/h (first-order in-basin decay)
    contact_time: float = 0.5  # hours (basin HRT)
    cl2_demand_per_tss: float = 0.02  # mg Cl2 / mg/L TSS demand

    # Sensors
    tmp_noise_std: float = 500.0
    tmp_lag: float = 0.9
    cl2_noise_std: float = 0.05
    cl2_lag: float = 0.8
    sensor_sample_period: int = 5

    # Reward
    reward_w_cl2_low: float = 2.0  # penalty for residual < target
    reward_w_cl2_high: float = 0.5  # penalty for residual > ceiling
    reward_w_tmp: float = 1.0
    reward_w_coag_cost: float = 0.05
    reward_w_cl2_cost: float = 0.1
    reward_w_throughput: float = 0.3
    cl2_target: float = 0.5  # mg/L minimum residual
    cl2_ceiling: float = 2.0  # mg/L max residual (taste/DBP)
    tmp_alarm: float = 1.5e5  # Pa


def make_drinking_water_train_benchmark(config: DrinkingWaterTrainConfig):
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
    turb_channel = ChannelParams(
        mean=config.turb_mean,
        amplitude=config.turb_amplitude,
        min_value=config.turb_min,
        max_value=config.turb_max,
        drift_scale=config.turb_drift_scale,
        drift_clip=config.turb_amplitude * 2.0,
    )
    tmp_analyzer = ResidualAnalyzerParams(
        noise_std=config.tmp_noise_std,
        lag_coefficient=config.tmp_lag,
        sample_period=config.sensor_sample_period,
    )
    cl2_analyzer = ResidualAnalyzerParams(
        noise_std=config.cl2_noise_std,
        lag_coefficient=config.cl2_lag,
        sample_period=config.sensor_sample_period,
    )

    dt = jnp.array(config.dt)
    dose_max = jnp.array(config.coag_dose_max)
    cl2_max = jnp.array(config.cl2_dose_max)
    flux_max = jnp.array(config.flux_max)
    max_tmp = jnp.array(config.membrane.max_tmp)
    mean_flow = jnp.array(config.mean_flow)
    area = jnp.array(config.membrane.area)

    def reset(rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)
        src = source_reset(k1)
        mem = membrane_reset(config.membrane, k2)
        state = DrinkingWaterTrainState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            turbidity_drift=jnp.array(0.0),
            membrane_state=mem,
            cl2_residual=jnp.array(1.0),
            tmp_sensor=analyzer_reset(k3),
            cl2_sensor=analyzer_reset(k4),
            last_q_in=jnp.array(config.mean_flow),
        )
        obs = jnp.zeros(8)
        return state, obs

    def step(state: DrinkingWaterTrainState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Raw water flow
        new_source, _transport, q_in, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Raw water turbidity
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        turb_signal = jnp.sin(2.0 * jnp.pi * t)
        new_turb_drift, raw_turbidity = channel_step(
            state.turbidity_drift,
            turb_signal,
            turb_channel,
            k2,
        )

        # 3. Coagulation stage
        coag_dose = jnp.clip(action[0], 0.0, 1.0) * dose_max
        # Approximate TSS from turbidity (rough NTU → mg/L)
        raw_tss = raw_turbidity * 2.5
        post_coag_tss, coag_eta = coagulate(raw_tss, raw_turbidity, coag_dose, config.coag)

        # 4. Membrane filtration
        flux = flux_max * 0.7  # fixed operating flux (could be action)
        do_backwash = action[1]
        # Air scour at constant 50% (could be action)
        air_scour = jnp.array(0.5)

        new_mem, tmp, permeate_tss, q_permeate = membrane_step(
            state.membrane_state,
            post_coag_tss,
            flux,
            air_scour,
            do_backwash,
            config.membrane,
            dt,
        )

        # 5. Chlorine disinfection
        cl2_dose = jnp.clip(action[2], 0.0, 1.0) * cl2_max
        # Chlorine demand depends on permeate quality
        cl2_demand = permeate_tss * config.cl2_demand_per_tss
        # Simple contact basin: exponential decay with demand
        applied_cl2 = cl2_dose - cl2_demand
        new_cl2_residual = jnp.maximum(
            0.0,
            applied_cl2 * jnp.exp(-config.cl2_decay_rate * config.contact_time),
        )

        # 6. Sensors
        new_tmp_sensor, sensed_tmp = analyzer_step(state.tmp_sensor, tmp, tmp_analyzer, k3)
        new_cl2_sensor, sensed_cl2 = analyzer_step(state.cl2_sensor, new_cl2_residual, cl2_analyzer, k4)

        # 7. Observation
        obs = jnp.array(
            [
                raw_turbidity / 50.0,
                post_coag_tss / 100.0,
                sensed_tmp / max_tmp,
                q_permeate / (flux_max * area),
                sensed_cl2 / 5.0,
                new_mem.hours_since_bw,
                coag_dose / dose_max,
                q_in / mean_flow,
            ]
        )

        # 8. Reward
        cl2_low = jnp.maximum(config.cl2_target - sensed_cl2, 0.0)
        cl2_high = jnp.maximum(sensed_cl2 - config.cl2_ceiling, 0.0)
        tmp_violation = jnp.maximum(sensed_tmp - config.tmp_alarm, 0.0)

        reward = (
            config.reward_w_throughput * (q_permeate / (flux_max * area))
            - config.reward_w_cl2_low * cl2_low**2
            - config.reward_w_cl2_high * cl2_high**2
            - config.reward_w_tmp * (tmp_violation / max_tmp) ** 2
            - config.reward_w_coag_cost * (coag_dose / dose_max)
            - config.reward_w_cl2_cost * (cl2_dose / cl2_max)
        )

        new_state = DrinkingWaterTrainState(
            step_count=state.step_count + 1,
            source_state=new_source,
            turbidity_drift=new_turb_drift,
            membrane_state=new_mem,
            cl2_residual=new_cl2_residual,
            tmp_sensor=new_tmp_sensor,
            cl2_sensor=new_cl2_sensor,
            last_q_in=q_in,
        )
        info: dict[str, jax.Array] = {
            "raw_turbidity": raw_turbidity,
            "post_coag_tss": post_coag_tss,
            "coag_eta": coag_eta,
            "tmp": tmp,
            "cl2_residual": new_cl2_residual,
            "q_permeate": q_permeate,
            "coag_dose": coag_dose,
            "cl2_dose": cl2_dose,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
