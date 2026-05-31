"""Primary clarifier control benchmark.

The agent controls primary sludge wastage rate to balance effluent TSS
quality against sludge inventory management.

Action (1D): normalised waste sludge rate [Q_w / Q_w_max]
Observation (5D):
  effluent TSS / 100
  underflow TSS / 10000
  sludge inventory / max_inventory
  flow / mean_flow
  dQ/dt
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.ramp_limited import RampLimitedActuatorParams, RampLimitedActuatorState
from process_control.actuators.ramp_limited import reset as actuator_reset
from process_control.actuators.ramp_limited import step as actuator_step
from process_control.scenarios.diurnal_source import DiurnalSourceParams, DiurnalSourceState
from process_control.scenarios.diurnal_source import reset as source_reset
from process_control.scenarios.diurnal_source import step as source_step
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.primary_clarifier import PrimaryClarifierParams, PrimaryClarifierState
from process_control.units.primary_clarifier import reset as clarifier_reset
from process_control.units.primary_clarifier import step as clarifier_step


@jax_dataclass
class PrimaryClarifierSensorState:
    eff_tss: ResidualAnalyzerState
    und_tss: ResidualAnalyzerState


@jax_dataclass
class PrimaryClarifierBenchState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    tss_drift: jax.Array
    clarifier_state: PrimaryClarifierState
    pump_state: RampLimitedActuatorState
    sensors: PrimaryClarifierSensorState
    last_q_in: jax.Array


@dataclass(frozen=True)
class PrimaryClarifierConfig:
    dt: float = 0.02
    steps_per_day: int = 1200

    # Influent flow
    mean_flow: float = 600.0
    diurnal_amplitude: float = 150.0
    min_flow: float = 350.0
    max_flow: float = 900.0
    drift_scale: float = 0.2

    # Influent TSS channel
    tss_mean: float = 250.0
    tss_amplitude: float = 80.0
    tss_min: float = 100.0
    tss_max: float = 500.0
    tss_drift_scale: float = 0.3

    # Clarifier
    clarifier: PrimaryClarifierParams = PrimaryClarifierParams()

    # Waste pump
    q_w_min: float = 5.0
    q_w_max: float = 80.0
    q_w_ramp_rate: float = 30.0

    # Sensors
    analyzer_noise_std: float = 5.0
    analyzer_lag: float = 0.7
    analyzer_sample_period: int = 5

    # Reward
    reward_w_eff: float = 1.0
    reward_w_inventory: float = 0.5
    reward_w_pump: float = 0.01
    eff_tss_limit: float = 80.0


def make_primary_clarifier_benchmark(config: PrimaryClarifierConfig):
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
    tss_channel_params = ChannelParams(
        mean=config.tss_mean,
        amplitude=config.tss_amplitude,
        min_value=config.tss_min,
        max_value=config.tss_max,
        drift_scale=config.tss_drift_scale,
        drift_clip=config.tss_amplitude,
    )
    pump_params = RampLimitedActuatorParams(
        max_output=config.q_w_max,
        min_output=config.q_w_min,
        max_ramp_rate=config.q_w_ramp_rate,
    )
    analyzer_params = ResidualAnalyzerParams(
        noise_std=config.analyzer_noise_std,
        lag_coefficient=config.analyzer_lag,
        sample_period=config.analyzer_sample_period,
    )

    dt = jnp.array(config.dt)
    q_mean = jnp.array(config.mean_flow)
    q_w_max = jnp.array(config.q_w_max)
    inv_max = jnp.array(config.clarifier.max_sludge_mass)
    eff_limit = jnp.array(config.eff_tss_limit)

    def reset(rng_key: jax.Array):
        k1, k2, k3 = jax.random.split(rng_key, 3)
        src = source_reset(k1)
        clar = clarifier_reset(config.clarifier, k2)
        pump = actuator_reset(k3)
        sensors = PrimaryClarifierSensorState(
            eff_tss=ResidualAnalyzerState.create(),
            und_tss=ResidualAnalyzerState.create(),
        )
        state = PrimaryClarifierBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            source_state=src,
            tss_drift=jnp.array(0.0),
            clarifier_state=clar,
            pump_state=pump,
            sensors=sensors,
            last_q_in=jnp.array(config.mean_flow),
        )
        obs = jnp.array(
            [
                config.tss_mean * 0.35 / 100.0,
                config.clarifier.min_underflow_tss / 10000.0,
                0.3,
                1.0,
                0.0,
            ]
        )
        return state, obs

    def step(state: PrimaryClarifierBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        # 1. Flow source
        new_source, _transport, flow, _demand = source_step(
            state.source_state,
            state.step_count,
            source_params,
            k1,
        )

        # 2. Influent TSS
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        tss_signal = jnp.sin(2.0 * jnp.pi * t)
        new_tss_drift, feed_tss = channel_step(state.tss_drift, tss_signal, tss_channel_params, k2)

        # 3. Waste pump actuator
        new_pump, q_waste = actuator_step(state.pump_state, action[0], pump_params, dt)

        # 4. Clarifier
        new_clar, eff_tss, und_tss = clarifier_step(
            state.clarifier_state,
            feed_tss,
            flow,
            q_waste,
            config.clarifier,
            dt,
        )

        # 5. Sensors
        new_eff_sensor, sensed_eff = analyzer_step(state.sensors.eff_tss, eff_tss, analyzer_params, k3)
        new_und_sensor, sensed_und = analyzer_step(state.sensors.und_tss, und_tss, analyzer_params, k4)

        # 6. Observation
        dq = (flow - state.last_q_in) / q_mean
        obs = jnp.array(
            [
                sensed_eff / 100.0,
                sensed_und / 10000.0,
                new_clar.sludge_mass / inv_max,
                flow / q_mean,
                dq,
            ]
        )

        # 7. Reward
        eff_violation = jnp.maximum(sensed_eff - eff_limit, 0.0)
        inv_deviation = jnp.abs(new_clar.sludge_mass / inv_max - 0.5)
        reward = -(config.reward_w_eff * (eff_violation / eff_limit) ** 2 + config.reward_w_inventory * inv_deviation**2 + config.reward_w_pump * (q_waste / q_w_max))

        new_state = PrimaryClarifierBenchState(
            step_count=state.step_count + 1,
            source_state=new_source,
            tss_drift=new_tss_drift,
            clarifier_state=new_clar,
            pump_state=new_pump,
            sensors=PrimaryClarifierSensorState(eff_tss=new_eff_sensor, und_tss=new_und_sensor),
            last_q_in=flow,
        )
        info: dict[str, jax.Array] = {
            "eff_tss": eff_tss,
            "und_tss": und_tss,
            "feed_tss": feed_tss,
            "q_waste": q_waste,
            "flow": flow,
            "sludge_mass": new_clar.sludge_mass,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
