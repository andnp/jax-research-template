from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class RampLimitedActuatorParams:
    max_output: float
    min_output: float
    max_ramp_rate: float | None = None
    max_ramp_up: float | None = None
    max_ramp_down: float | None = None


@jax_dataclass
class RampLimitedActuatorState:
    current_output: jax.Array

    @staticmethod
    def create() -> "RampLimitedActuatorState":
        return RampLimitedActuatorState(current_output=jnp.array(0.0))


@jax_dataclass
class RampLimitedActuatorOutput:
    """Actuator output with saturation and rate-limit diagnostics."""

    requested: jax.Array
    realized: jax.Array
    is_saturated: jax.Array
    is_rate_limited: jax.Array


def reset(rng_key: jax.Array) -> RampLimitedActuatorState:
    return RampLimitedActuatorState.create()


def step_with_diagnostics(
    state: RampLimitedActuatorState,
    requested: jax.Array,
    params: RampLimitedActuatorParams,
    dt: jax.Array,
) -> tuple[RampLimitedActuatorState, RampLimitedActuatorOutput]:
    """Apply bounds and asymmetric rate limits while reporting activation."""
    ramp_up = params.max_ramp_up
    ramp_down = params.max_ramp_down
    if params.max_ramp_rate is not None:
        ramp_up = params.max_ramp_rate if ramp_up is None else ramp_up
        ramp_down = params.max_ramp_rate if ramp_down is None else ramp_down
    if ramp_up is None:
        ramp_up = jnp.inf
    if ramp_down is None:
        ramp_down = jnp.inf

    bounded = jnp.clip(requested, params.min_output, params.max_output)
    upper = state.current_output + ramp_up * dt
    lower = state.current_output - ramp_down * dt
    rate_limited = jnp.minimum(jnp.maximum(bounded, lower), upper)
    realized = jnp.clip(rate_limited, params.min_output, params.max_output)
    result = RampLimitedActuatorOutput(
        requested=requested,
        realized=realized,
        is_saturated=requested != bounded,
        is_rate_limited=rate_limited != bounded,
    )
    return RampLimitedActuatorState(current_output=realized), result


def step(
    state: RampLimitedActuatorState,
    requested: jax.Array,
    params: RampLimitedActuatorParams,
    dt: jax.Array,
) -> tuple[RampLimitedActuatorState, jax.Array]:
    """Apply bounds and rate limits, returning only the realized output."""
    new_state, result = step_with_diagnostics(state, requested, params, dt)
    return new_state, result.realized
