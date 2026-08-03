"""Small JAX-native PID controller used by dosing loops."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PIDControllerParams:
    """PID gains and output limits."""

    kp: float
    ki: float
    kd: float
    ff: float
    output_min: float
    output_max: float
    max_integral: float


@jax_dataclass
class PIDControllerState:
    """Persistent PID integral and derivative memory."""

    integral: jax.Array
    previous_measurement: jax.Array
    initialized: jax.Array

    @staticmethod
    def create() -> "PIDControllerState":
        return PIDControllerState(
            integral=jnp.array(0.0),
            previous_measurement=jnp.array(0.0),
            initialized=jnp.array(False),
        )


@jax_dataclass
class PIDControllerOutput:
    """PID output before and after actuator saturation."""

    raw: jax.Array
    saturated: jax.Array
    is_saturated: jax.Array


def step_with_diagnostics(
    state: PIDControllerState,
    measurement: jax.Array,
    setpoint: jax.Array,
    params: PIDControllerParams,
    dt: jax.Array,
) -> tuple[PIDControllerState, PIDControllerOutput]:
    """Advance a measurement-derivative PID controller."""
    error = setpoint - measurement
    candidate_integral = jnp.clip(
        state.integral + error * dt,
        -params.max_integral,
        params.max_integral,
    )
    derivative = jnp.where(
        state.initialized,
        -(measurement - state.previous_measurement) / dt,
        0.0,
    )
    candidate_output = (
        params.kp * error
        + params.ki * candidate_integral
        + params.kd * derivative
        + params.ff
    )
    integral_output_change = params.ki * error * dt
    drives_further_high = (candidate_output > params.output_max) & (
        integral_output_change > 0.0
    )
    drives_further_low = (candidate_output < params.output_min) & (
        integral_output_change < 0.0
    )
    new_integral = jnp.where(
        drives_further_high | drives_further_low,
        state.integral,
        candidate_integral,
    )
    raw_output = (
        params.kp * error
        + params.ki * new_integral
        + params.kd * derivative
        + params.ff
    )
    return (
        PIDControllerState(
            integral=new_integral,
            previous_measurement=measurement,
            initialized=jnp.array(True),
        ),
        PIDControllerOutput(
            raw=raw_output,
            saturated=jnp.clip(raw_output, params.output_min, params.output_max),
            is_saturated=raw_output != jnp.clip(
                raw_output,
                params.output_min,
                params.output_max,
            ),
        ),
    )
