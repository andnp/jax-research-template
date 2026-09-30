from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PIControllerParams:
    kp: float | jax.Array
    ki: float | jax.Array
    ff: float | jax.Array
    output_min: float | jax.Array
    output_max: float | jax.Array
    max_integral: float | jax.Array | None = None


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


def integral_limit(params: PIControllerParams):
    """Anti-windup clamp: explicit `max_integral`, else the smallest bound letting the integral term span the output range from `ff`.

    The guarantee covers the PI output itself; a delta added after saturation
    (FEEDFORWARD dosing) is outside it. Zero `ki` has no integral authority to
    size, so the integral is pinned at zero.
    """
    if params.max_integral is not None:
        return params.max_integral
    authority = jnp.maximum(params.output_max - params.ff, params.ff - params.output_min)
    has_integral_action = params.ki != 0.0
    return jnp.where(has_integral_action, authority / jnp.where(has_integral_action, jnp.abs(params.ki), 1.0), 0.0)


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
    max_integral = integral_limit(params)
    candidate_integral = jnp.clip(state.integral + error * dt, -max_integral, max_integral)
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
    max_integral = integral_limit(params)
    return PIControllerState(integral=jnp.clip(tracked_integral, -max_integral, max_integral))
