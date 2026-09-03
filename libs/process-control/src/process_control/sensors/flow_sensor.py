from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class FlowSensorParams:
    noise_std: float
    bias: float
    dropout_probability: float


@jax_dataclass
class FlowSensorState:
    held_value: jax.Array

    @staticmethod
    def create() -> "FlowSensorState":
        return FlowSensorState(held_value=jnp.array(0.0))


def reset(rng_key: jax.Array) -> FlowSensorState:
    return FlowSensorState.create()


def step(state: FlowSensorState, true_flow: jax.Array, params: FlowSensorParams, rng_key: jax.Array) -> tuple[FlowSensorState, jax.Array]:
    k1, k2 = jax.random.split(rng_key, 2)
    noise = jax.random.normal(k1) * params.noise_std
    measured = true_flow + noise + params.bias

    dropout = jax.random.uniform(k2) < params.dropout_probability
    value = jnp.where(dropout, state.held_value, measured)
    return FlowSensorState(held_value=value), value
