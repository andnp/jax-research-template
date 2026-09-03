from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PhSensorParams:
    noise_std: float
    lag_coefficient: float


@jax_dataclass
class PhSensorState:
    held_value: jax.Array

    @staticmethod
    def create() -> "PhSensorState":
        return PhSensorState(held_value=jnp.array(7.0))


def reset(rng_key: jax.Array) -> PhSensorState:
    return PhSensorState.create()


def step(
    state: PhSensorState,
    true_ph: jax.Array,
    params: PhSensorParams,
    rng_key: jax.Array,
) -> tuple[PhSensorState, jax.Array]:
    noise = jax.random.normal(rng_key) * params.noise_std
    raw = true_ph + noise
    smoothed = params.lag_coefficient * state.held_value + (1.0 - params.lag_coefficient) * raw
    return PhSensorState(held_value=smoothed), smoothed
