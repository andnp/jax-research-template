from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class RampLimitedActuatorParams:
    max_output: float
    min_output: float
    max_ramp_rate: float


@jax_dataclass
class RampLimitedActuatorState:
    current_output: jax.Array

    @staticmethod
    def create() -> "RampLimitedActuatorState":
        return RampLimitedActuatorState(current_output=jnp.array(0.0))


def reset(rng_key: jax.Array) -> RampLimitedActuatorState:
    return RampLimitedActuatorState.create()


def step(state: RampLimitedActuatorState, requested: jax.Array, params: RampLimitedActuatorParams, dt: jax.Array) -> tuple[RampLimitedActuatorState, jax.Array]:
    max_change = params.max_ramp_rate * dt
    delta = jnp.clip(requested - state.current_output, -max_change, max_change)
    new_output = jnp.clip(state.current_output + delta, params.min_output, params.max_output)
    return RampLimitedActuatorState(current_output=new_output), new_output
