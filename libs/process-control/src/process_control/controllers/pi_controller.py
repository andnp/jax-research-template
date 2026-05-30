from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PIControllerParams:
    kp: float
    ki: float
    ff: float
    output_min: float
    output_max: float
    max_integral: float


@jax_dataclass
class PIControllerState:
    integral: jax.Array

    @staticmethod
    def create() -> "PIControllerState":
        return PIControllerState(integral=jnp.array(0.0))


def reset(rng_key: jax.Array) -> PIControllerState:
    return PIControllerState.create()


def step(
    state: PIControllerState,
    measurement: jax.Array,
    setpoint: jax.Array,
    params: PIControllerParams,
    dt: jax.Array,
) -> tuple[PIControllerState, jax.Array]:
    error = setpoint - measurement
    new_integral = jnp.clip(state.integral + error * dt, -params.max_integral, params.max_integral)
    output = params.kp * error + params.ki * new_integral + params.ff
    output = jnp.clip(output, params.output_min, params.output_max)
    return PIControllerState(integral=new_integral), output
