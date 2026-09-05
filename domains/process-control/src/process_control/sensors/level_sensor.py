from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class LevelSensorParams:
    noise_std: float
    lag_coefficient: float


@jax_dataclass
class LevelSensorState:
    held_value: jax.Array

    @staticmethod
    def create(initial_level: float) -> "LevelSensorState":
        return LevelSensorState(held_value=jnp.array(initial_level))


def reset(initial_level: float, rng_key: jax.Array) -> LevelSensorState:
    return LevelSensorState.create(initial_level)


def step(
    state: LevelSensorState,
    true_level: jax.Array,
    params: LevelSensorParams,
    rng_key: jax.Array,
) -> tuple[LevelSensorState, jax.Array]:
    noise = jax.random.normal(rng_key) * params.noise_std
    raw = true_level + noise
    smoothed = params.lag_coefficient * state.held_value + (1.0 - params.lag_coefficient) * raw
    return LevelSensorState(held_value=smoothed), smoothed
