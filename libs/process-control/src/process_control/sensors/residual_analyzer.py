from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class ResidualAnalyzerParams:
    noise_std: float
    lag_coefficient: float
    sample_period: int


@jax_dataclass
class ResidualAnalyzerState:
    held_value: jax.Array
    steps_since_sample: jax.Array

    @staticmethod
    def create() -> "ResidualAnalyzerState":
        return ResidualAnalyzerState(
            held_value=jnp.array(0.0),
            steps_since_sample=jnp.array(0, dtype=jnp.int32),
        )


def reset(rng_key: jax.Array) -> ResidualAnalyzerState:
    return ResidualAnalyzerState.create()


def step(
    state: ResidualAnalyzerState,
    true_residual: jax.Array,
    params: ResidualAnalyzerParams,
    rng_key: jax.Array,
) -> tuple[ResidualAnalyzerState, jax.Array]:
    should_sample = state.steps_since_sample >= params.sample_period

    noise = jax.random.normal(rng_key) * params.noise_std
    raw_measurement = true_residual + noise

    smoothed = params.lag_coefficient * state.held_value + (1.0 - params.lag_coefficient) * raw_measurement

    new_value = jnp.where(should_sample, smoothed, state.held_value)
    new_steps = jnp.where(should_sample, jnp.array(0, dtype=jnp.int32), state.steps_since_sample + 1)

    return ResidualAnalyzerState(held_value=new_value, steps_since_sample=new_steps), new_value
