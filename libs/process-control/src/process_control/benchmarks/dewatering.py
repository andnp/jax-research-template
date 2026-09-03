"""Sludge dewatering control benchmark.

The agent controls polymer dose and belt speed to optimise cake dryness
while minimising polymer cost and maintaining acceptable filtrate quality.

Action (2D): [polymer_dose / dose_max, belt_speed (0-1)]
Observation (5D):
  cake_dryness (fraction)
  filtrate_tss / 500
  throughput / max_throughput
  polymer_dose / dose_max
  feed_tss / 15000
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.dewatering import DewateringParams, DewateringState
from process_control.units.dewatering import reset as dewatering_reset
from process_control.units.dewatering import step as dewatering_step


@jax_dataclass
class DewateringBenchState:
    step_count: jax.Array
    dewatering_state: DewateringState
    tss_drift: jax.Array
    filtrate_sensor: ResidualAnalyzerState
    last_feed_tss: jax.Array


@dataclass(frozen=True)
class DewateringConfig:
    dt: float = 0.05  # hours per step (~3 min)
    steps_per_day: int = 480

    # Dewatering unit
    dewatering: DewateringParams = DewateringParams()

    # Feed sludge TSS variation
    feed_tss_mean: float = 10000.0  # g/m³ (1% solids)
    feed_tss_amplitude: float = 3000.0
    feed_tss_min: float = 5000.0
    feed_tss_max: float = 18000.0
    feed_tss_drift_scale: float = 0.3

    # Feed flow (constant)
    q_feed: float = 30.0  # m³/h

    # Polymer
    dose_max: float = 15.0  # mg/L max polymer dose

    # Sensor
    filtrate_noise_std: float = 10.0
    filtrate_lag: float = 0.6
    filtrate_sample_period: int = 3

    # Reward
    target_dryness: float = 0.25
    reward_w_dryness: float = 1.0
    reward_w_cost: float = 0.05
    reward_w_filtrate: float = 0.3
    filtrate_limit: float = 200.0  # g/m³


def make_dewatering_benchmark(config: DewateringConfig):
    tss_channel_params = ChannelParams(
        mean=config.feed_tss_mean,
        amplitude=config.feed_tss_amplitude,
        min_value=config.feed_tss_min,
        max_value=config.feed_tss_max,
        drift_scale=config.feed_tss_drift_scale,
        drift_clip=config.feed_tss_amplitude,
    )
    filtrate_analyzer_params = ResidualAnalyzerParams(
        noise_std=config.filtrate_noise_std,
        lag_coefficient=config.filtrate_lag,
        sample_period=config.filtrate_sample_period,
    )

    dt = jnp.array(config.dt)
    dose_max = jnp.array(config.dose_max)
    q_feed = jnp.array(config.q_feed)
    target_dryness = jnp.array(config.target_dryness)
    filtrate_limit = jnp.array(config.filtrate_limit)

    def reset(rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)
        dw = dewatering_reset(config.dewatering, k1)
        filt_sensor = analyzer_reset(k2)
        state = DewateringBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            dewatering_state=dw,
            tss_drift=jnp.array(0.0),
            filtrate_sensor=filt_sensor,
            last_feed_tss=jnp.array(config.feed_tss_mean),
        )
        obs = jnp.array(
            [
                config.dewatering.base_dryness,
                0.0,
                0.5,
                0.0,
                config.feed_tss_mean / 15000.0,
            ]
        )
        return state, obs

    def step(state: DewateringBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)

        # 1. Feed TSS variation
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        tss_signal = jnp.sin(2.0 * jnp.pi * t)
        new_tss_drift, feed_tss = channel_step(state.tss_drift, tss_signal, tss_channel_params, k1)

        # 2. Actions
        polymer_dose = jnp.clip(action[0], 0.0, 1.0) * dose_max
        belt_speed = jnp.clip(action[1], 0.0, 1.0)

        # 3. Dewatering step
        new_dw, dryness, filtrate_tss, q_filtrate = dewatering_step(
            state.dewatering_state,
            feed_tss,
            q_feed,
            polymer_dose,
            belt_speed,
            config.dewatering,
            dt,
        )

        # 4. Filtrate sensor
        new_filt_sensor, sensed_filtrate = analyzer_step(
            state.filtrate_sensor,
            filtrate_tss,
            filtrate_analyzer_params,
            k2,
        )

        # 5. Observation
        throughput = jnp.minimum(q_feed, belt_speed * config.dewatering.max_throughput)
        obs = jnp.array(
            [
                dryness,
                sensed_filtrate / 500.0,
                throughput / config.dewatering.max_throughput,
                polymer_dose / dose_max,
                feed_tss / 15000.0,
            ]
        )

        # 6. Reward
        dryness_gap = jnp.maximum(target_dryness - dryness, 0.0)
        filtrate_violation = jnp.maximum(sensed_filtrate - filtrate_limit, 0.0)
        reward = -(
            config.reward_w_dryness * dryness_gap**2 + config.reward_w_cost * (polymer_dose / dose_max) + config.reward_w_filtrate * (filtrate_violation / filtrate_limit) ** 2
        )

        new_state = DewateringBenchState(
            step_count=state.step_count + 1,
            dewatering_state=new_dw,
            tss_drift=new_tss_drift,
            filtrate_sensor=new_filt_sensor,
            last_feed_tss=feed_tss,
        )
        info: dict[str, jax.Array] = {
            "dryness": dryness,
            "filtrate_tss": filtrate_tss,
            "polymer_dose": polymer_dose,
            "feed_tss": feed_tss,
            "throughput": throughput,
            "q_filtrate": q_filtrate,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
