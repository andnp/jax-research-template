from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class LaggedSensorParams:
    noise_std: float
    lag_coefficient: float
    sample_period: int = 0
    drift_rate: float = 0.0


@jax_dataclass
class LaggedSensorState:
    held_value: jax.Array
    steps_since_sample: jax.Array
    drift_bias: jax.Array

    @staticmethod
    def create(initial_value: float | jax.Array) -> "LaggedSensorState":
        return LaggedSensorState(
            held_value=jnp.asarray(initial_value),
            steps_since_sample=jnp.array(0, dtype=jnp.int32),
            drift_bias=jnp.array(0.0),
        )


def reset(initial_value: float | jax.Array, rng_key: jax.Array) -> LaggedSensorState:
    del rng_key
    return LaggedSensorState.create(initial_value)


def step(
    state: LaggedSensorState,
    true_value: jax.Array,
    params: LaggedSensorParams,
    rng_key: jax.Array,
) -> tuple[LaggedSensorState, jax.Array]:
    if params.drift_rate > 0.0:
        drift_key, noise_key = jax.random.split(rng_key)
        new_drift = state.drift_bias + jax.random.normal(drift_key) * params.drift_rate
    else:
        noise_key = rng_key
        new_drift = state.drift_bias

    raw_measurement = true_value + jax.random.normal(noise_key) * params.noise_std + new_drift
    smoothed = params.lag_coefficient * state.held_value + (1.0 - params.lag_coefficient) * raw_measurement

    if params.sample_period > 0:
        should_sample = state.steps_since_sample >= params.sample_period
        new_value = jnp.where(should_sample, smoothed, state.held_value)
        new_steps = jnp.where(should_sample, 0, state.steps_since_sample + 1)
    else:
        new_value = smoothed
        new_steps = jnp.array(0, dtype=jnp.int32)

    return LaggedSensorState(new_value, new_steps, new_drift), new_value
