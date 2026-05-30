from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DosePumpParams:
    max_dose: float
    min_dose: float
    max_ramp_rate: float


@dataclass(frozen=True)
class DosePumpState:
    current_output: jax.Array

    @staticmethod
    def create() -> "DosePumpState":
        return DosePumpState(current_output=jnp.array(0.0))


jax.tree_util.register_dataclass(
    DosePumpState,
    data_fields=["current_output"],
    meta_fields=[],
)


def reset(rng_key: jax.Array) -> DosePumpState:
    return DosePumpState.create()


def step(state: DosePumpState, requested_dose: jax.Array, params: DosePumpParams, dt: jax.Array) -> tuple[DosePumpState, jax.Array]:
    max_change = params.max_ramp_rate * dt
    delta = jnp.clip(requested_dose - state.current_output, -max_change, max_change)
    new_output = jnp.clip(state.current_output + delta, params.min_dose, params.max_dose)
    return DosePumpState(current_output=new_output), new_output
