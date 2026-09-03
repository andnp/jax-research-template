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


@jax_dataclass
class PIControllerOutput:
    """PI output before and after actuator saturation."""

    raw: jax.Array
    saturated: jax.Array
    is_saturated: jax.Array


def reset(rng_key: jax.Array) -> PIControllerState:
    return PIControllerState.create()


def step(
    state: PIControllerState,
    measurement: jax.Array,
    setpoint: jax.Array,
    params: PIControllerParams,
    dt: jax.Array,
) -> tuple[PIControllerState, jax.Array]:
    """Advance the PI controller using conditional-integration anti-windup."""
    new_state, result = step_with_diagnostics(state, measurement, setpoint, params, dt)
    return new_state, result.saturated


def step_with_diagnostics(
    state: PIControllerState,
    measurement: jax.Array,
    setpoint: jax.Array,
    params: PIControllerParams,
    dt: jax.Array,
) -> tuple[PIControllerState, PIControllerOutput]:
    """Advance the PI controller and report raw and saturated outputs."""
    error = setpoint - measurement
    candidate_integral = jnp.clip(
        state.integral + error * dt,
        -params.max_integral,
        params.max_integral,
    )
    candidate_output = params.kp * error + params.ki * candidate_integral + params.ff

    integral_output_change = params.ki * error * dt
    drives_further_high = (candidate_output > params.output_max) & (integral_output_change > 0.0)
    drives_further_low = (candidate_output < params.output_min) & (integral_output_change < 0.0)
    new_integral = jnp.where(
        drives_further_high | drives_further_low,
        state.integral,
        candidate_integral,
    )
    raw_output = params.kp * error + params.ki * new_integral + params.ff
    saturated_output = jnp.clip(raw_output, params.output_min, params.output_max)
    result = PIControllerOutput(
        raw=raw_output,
        saturated=saturated_output,
        is_saturated=raw_output != saturated_output,
    )
    return PIControllerState(integral=new_integral), result


def track_output(
    state: PIControllerState,
    measurement: jax.Array,
    setpoint: jax.Array,
    output: jax.Array,
    params: PIControllerParams,
) -> PIControllerState:
    """Track a manual output so a later automatic transfer is bumpless."""
    error = setpoint - measurement
    has_integral_action = params.ki != 0.0
    safe_ki = jnp.where(has_integral_action, params.ki, 1.0)
    tracked_integral = jnp.where(
        has_integral_action,
        (jnp.clip(output, params.output_min, params.output_max) - params.ff - params.kp * error) / safe_ki,
        state.integral,
    )
    return PIControllerState(integral=jnp.clip(tracked_integral, -params.max_integral, params.max_integral))
