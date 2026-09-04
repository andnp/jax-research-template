"""Bounded PI feedback composed with the shared ramp-limited actuator."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators.ramp_limited import (
    RampLimitedActuatorParams,
    RampLimitedActuatorState,
)
from process_control.actuators.ramp_limited import (
    step_with_diagnostics as actuator_step_with_diagnostics,
)


@dataclass(frozen=True)
class BoundedPIControllerParams:
    """PI gains, bounds, anti-windup, and actuator-rate settings."""

    kp: float
    ki: float
    ff: float
    output_min: float
    output_max: float
    dt: float
    anti_windup_gain: float = 1.0
    max_ramp_up: float | None = None
    max_ramp_down: float | None = None


@jax_dataclass
class BoundedPIControllerState:
    """Persistent integral and realized-output memory for bounded PI."""

    integral_error: jax.Array
    previous_output: jax.Array
    initialized: jax.Array

    @staticmethod
    def create(initial_output: jax.Array | float = 0.0) -> "BoundedPIControllerState":
        return BoundedPIControllerState(
            integral_error=jnp.zeros_like(jnp.asarray(initial_output)),
            previous_output=jnp.asarray(initial_output),
            initialized=jnp.zeros_like(jnp.asarray(initial_output), dtype=bool),
        )


@jax_dataclass
class BoundedPIControllerOutput:
    """Bounded PI request, realized action, and actuator diagnostics."""

    requested: jax.Array
    realized: jax.Array
    is_saturated: jax.Array
    is_rate_limited: jax.Array


def step(
    state: BoundedPIControllerState,
    measurement: jax.Array,
    setpoint: jax.Array,
    params: BoundedPIControllerParams,
    *,
    feedforward: jax.Array | float = 0.0,
    bumpless: bool = True,
) -> tuple[BoundedPIControllerState, BoundedPIControllerOutput]:
    """Advance bounded PI with back-calculation through the actuator."""
    error = setpoint - measurement
    previous = jnp.clip(state.previous_output, params.output_min, params.output_max)
    feedforward = params.ff + jnp.asarray(feedforward)
    safe_ki = jnp.where(params.ki != 0.0, params.ki, 1.0)
    bumpless_integral = (previous - feedforward - params.kp * error) / safe_ki
    integral = jnp.where(
        (~state.initialized) & (params.ki != 0.0) & bumpless,
        bumpless_integral,
        state.integral_error,
    )
    candidate_integral = jnp.where(
        state.initialized,
        integral + params.dt * error,
        integral,
    )
    requested = feedforward + params.kp * error + params.ki * candidate_integral
    actuator_params = RampLimitedActuatorParams(
        max_output=params.output_max,
        min_output=params.output_min,
        max_ramp_up=params.max_ramp_up,
        max_ramp_down=params.max_ramp_down,
    )
    _, limited = actuator_step_with_diagnostics(
        RampLimitedActuatorState(current_output=previous),
        requested,
        actuator_params,
        jnp.asarray(params.dt),
    )
    corrected_integral = candidate_integral + (
        params.dt
        * params.anti_windup_gain
        * (limited.realized - requested)
        / safe_ki
    )
    anti_windup = (params.ki != 0.0) & (params.anti_windup_gain != 0.0)
    candidate_integral = jnp.where(
        state.initialized & anti_windup,
        corrected_integral,
        candidate_integral,
    )
    requested = feedforward + params.kp * error + params.ki * candidate_integral
    _, limited = actuator_step_with_diagnostics(
        RampLimitedActuatorState(current_output=previous),
        requested,
        actuator_params,
        jnp.asarray(params.dt),
    )
    return (
        BoundedPIControllerState(
            integral_error=candidate_integral,
            previous_output=limited.realized,
            initialized=jnp.ones_like(state.initialized, dtype=bool),
        ),
        BoundedPIControllerOutput(
            requested=requested,
            realized=limited.realized,
            is_saturated=limited.is_saturated,
            is_rate_limited=limited.is_rate_limited,
        ),
    )
