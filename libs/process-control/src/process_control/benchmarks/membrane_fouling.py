"""Membrane fouling control benchmark.

The agent controls membrane flux, air scour, and backwash timing to
maintain throughput while managing fouling.

Action (3D): [flux / flux_max, air_scour (0-1), backwash_trigger (>0.5 = wash)]
Observation (6D):
  TMP / max_TMP
  permeability_ratio (current / clean)
  permeate_flow / max_flow
  air_scour
  hours_since_backwash / 1.0
  feed_tss / 1000
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState
from process_control.sensors.residual_analyzer import reset as analyzer_reset
from process_control.sensors.residual_analyzer import step as analyzer_step
from process_control.units.membrane import MembraneParams, MembraneState
from process_control.units.membrane import reset as membrane_reset
from process_control.units.membrane import step as membrane_step


@jax_dataclass
class MembraneFoulingBenchState:
    step_count: jax.Array
    membrane_state: MembraneState
    tss_drift: jax.Array
    tmp_sensor: ResidualAnalyzerState
    last_feed_tss: jax.Array


@dataclass(frozen=True)
class MembraneFoulingConfig:
    dt: float = 0.01  # hours (~36 sec, needs fine resolution for backwash)
    steps_per_day: int = 2400

    # Membrane
    membrane: MembraneParams = MembraneParams()

    # Operating range
    flux_max: float = 0.05  # m/h (50 LMH — typical UF)
    flux_min: float = 0.01  # m/h

    # Feed TSS variation
    feed_tss_mean: float = 50.0  # g/m³ (typical MF/UF feed)
    feed_tss_amplitude: float = 20.0
    feed_tss_min: float = 10.0
    feed_tss_max: float = 150.0
    feed_tss_drift_scale: float = 0.3

    # TMP sensor
    tmp_noise_std: float = 500.0  # Pa
    tmp_lag: float = 0.9
    tmp_sample_period: int = 5

    # Reward
    reward_w_tmp: float = 1.0  # penalty for high TMP
    reward_w_throughput: float = 0.5  # reward for permeate production
    reward_w_scour: float = 0.1  # penalty for air scour cost
    reward_w_bw: float = 0.3  # penalty for backwash (lost production)
    tmp_target: float = 0.8e5  # Pa — target TMP (0.8 bar)


def make_membrane_fouling_benchmark(config: MembraneFoulingConfig):
    tss_channel_params = ChannelParams(
        mean=config.feed_tss_mean,
        amplitude=config.feed_tss_amplitude,
        min_value=config.feed_tss_min,
        max_value=config.feed_tss_max,
        drift_scale=config.feed_tss_drift_scale,
        drift_clip=config.feed_tss_amplitude,
    )
    tmp_analyzer_params = ResidualAnalyzerParams(
        noise_std=config.tmp_noise_std,
        lag_coefficient=config.tmp_lag,
        sample_period=config.tmp_sample_period,
    )

    dt = jnp.array(config.dt)
    flux_max = jnp.array(config.flux_max)
    max_tmp = jnp.array(config.membrane.max_tmp)
    tmp_target = jnp.array(config.tmp_target)
    area = jnp.array(config.membrane.area)

    def reset(rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)
        mem = membrane_reset(config.membrane, k1)
        tmp_sensor = analyzer_reset(k2)
        state = MembraneFoulingBenchState(
            step_count=jnp.array(0, dtype=jnp.int32),
            membrane_state=mem,
            tss_drift=jnp.array(0.0),
            tmp_sensor=tmp_sensor,
            last_feed_tss=jnp.array(config.feed_tss_mean),
        )
        obs = jnp.zeros(6)
        return state, obs

    def step(state: MembraneFoulingBenchState, action: jax.Array, rng_key: jax.Array):
        k1, k2 = jax.random.split(rng_key, 2)

        # 1. Feed TSS
        t = (state.step_count % config.steps_per_day) / config.steps_per_day
        tss_signal = jnp.sin(2.0 * jnp.pi * t)
        new_tss_drift, feed_tss = channel_step(state.tss_drift, tss_signal, tss_channel_params, k1)

        # 2. Actions
        flux = jnp.clip(action[0], 0.0, 1.0) * flux_max
        air_scour = jnp.clip(action[1], 0.0, 1.0)
        do_backwash = action[2]

        # 3. Membrane step
        new_mem, tmp, permeate_tss, q_permeate = membrane_step(
            state.membrane_state,
            feed_tss,
            flux,
            air_scour,
            do_backwash,
            config.membrane,
            dt,
        )

        # 4. TMP sensor
        new_tmp_sensor, sensed_tmp = analyzer_step(state.tmp_sensor, tmp, tmp_analyzer_params, k2)

        # 5. Permeability ratio (clean vs current)
        r_total = config.membrane.r_membrane + new_mem.r_reversible + new_mem.r_irreversible
        perm_ratio = config.membrane.r_membrane / (r_total + 1e-10)

        # 6. Observation
        obs = jnp.array(
            [
                sensed_tmp / max_tmp,
                perm_ratio,
                q_permeate / (flux_max * area),
                air_scour,
                new_mem.hours_since_bw,
                feed_tss / 1000.0,
            ]
        )

        # 7. Reward: throughput benefit minus costs
        tmp_violation = jnp.maximum(sensed_tmp - tmp_target, 0.0)
        starts_bw = (
            (do_backwash > 0.5)
            & ~state.membrane_state.is_backwashing
            & ~state.membrane_state.backwash_trigger_latched
        )
        reward = (
            config.reward_w_throughput * (q_permeate / (flux_max * area))
            - config.reward_w_tmp * (tmp_violation / max_tmp) ** 2
            - config.reward_w_scour * air_scour
            - config.reward_w_bw * starts_bw
        )

        new_state = MembraneFoulingBenchState(
            step_count=state.step_count + 1,
            membrane_state=new_mem,
            tss_drift=new_tss_drift,
            tmp_sensor=new_tmp_sensor,
            last_feed_tss=feed_tss,
        )
        info: dict[str, jax.Array] = {
            "tmp": tmp,
            "perm_ratio": perm_ratio,
            "q_permeate": q_permeate,
            "feed_tss": feed_tss,
            "permeate_tss": permeate_tss,
            "r_reversible": new_mem.r_reversible,
            "r_irreversible": new_mem.r_irreversible,
        }
        done = jnp.array(False)
        return new_state, obs, reward, done, info

    return reset, step
