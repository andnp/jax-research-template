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

    kp: float | jax.Array
    ki: float | jax.Array
    ff: float | jax.Array
    output_min: float | jax.Array
    output_max: float | jax.Array
    dt: float | jax.Array
    anti_windup_gain: float | jax.Array = 1.0
    max_ramp_up: float | jax.Array | None = None
    max_ramp_down: float | jax.Array | None = None


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


def step_from_observation(
    observation: jax.Array,
    state: BoundedPIControllerState,
    *,
    kp: float | jax.Array,
    ki: float | jax.Array,
    ff: float | jax.Array,
    output_min: float | jax.Array,
    output_max: float | jax.Array,
    dt: float | jax.Array,
    anti_windup_gain: float | jax.Array = 1.0,
    max_ramp_up: float | jax.Array | None = None,
    max_ramp_down: float | jax.Array | None = None,
    measurement_index: int = 1,
    setpoint_index: int = 2,
    setpoint_override: jax.Array | None = None,
    feedforward_index: int | None = None,
    feedforward_gain: float | jax.Array = 0.0,
    bumpless: bool = True,
) -> tuple[jax.Array, BoundedPIControllerState]:
    """Apply bounded PI to measured and setpoint fields in an observation."""
    observation = jnp.asarray(observation)
    measured = observation[..., measurement_index]
    setpoint = (
        observation[..., setpoint_index]
        if setpoint_override is None
        else jnp.asarray(setpoint_override)
    )
    feedforward = jnp.asarray(0.0)
    if feedforward_index is not None:
        feedforward = feedforward + feedforward_gain * observation[..., feedforward_index]
    params = BoundedPIControllerParams(
        kp=kp,
        ki=ki,
        ff=ff,
        output_min=output_min,
        output_max=output_max,
        dt=dt,
        anti_windup_gain=anti_windup_gain,
        max_ramp_up=max_ramp_up,
        max_ramp_down=max_ramp_down,
    )
    updated_state, result = step(
        state,
        measured,
        setpoint,
        params,
        feedforward=feedforward,
        bumpless=bumpless,
    )
    return jnp.expand_dims(result.realized, axis=-1), updated_state


@dataclass(frozen=True, slots=True)
class BoundedPIController:
    """Bounded, rate-limited PI policy for measured observations."""

    kp: float
    ki: float
    ff: float = 0.0
    output_min: float = 0.0
    output_max: float = 1.0
    dt: float = 1.0
    anti_windup_gain: float = 1.0
    max_ramp_up: float | None = None
    max_ramp_down: float | None = None
    measurement_index: int = 1
    setpoint_index: int = 2
    error_index: int | None = None
    feedforward_index: int | None = None
    feedforward_gain: float = 0.0
    initial_output: float | None = None
    bumpless: bool = True

    def __post_init__(self) -> None:
        """Validate controller settings before an experiment starts."""
        if self.output_max <= self.output_min:
            raise ValueError("output_max must be greater than output_min")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.anti_windup_gain < 0.0:
            raise ValueError("anti_windup_gain must be non-negative")
        if self.max_ramp_up is not None and self.max_ramp_up <= 0.0:
            raise ValueError("max_ramp_up must be positive when provided")
        if self.max_ramp_down is not None and self.max_ramp_down <= 0.0:
            raise ValueError("max_ramp_down must be positive when provided")

    def reset(self, key: jax.Array) -> BoundedPIControllerState:
        """Reset integral memory and establish the initial output."""
        del key
        initial = self.ff if self.initial_output is None else self.initial_output
        initial = min(max(initial, self.output_min), self.output_max)
        return BoundedPIControllerState.create(initial)

    def act(
        self,
        observation: jax.Array,
        state: object,
        key: jax.Array,
    ) -> tuple[jax.Array, BoundedPIControllerState]:
        """Compute one bounded feedback action from the observed PV and SP."""
        del key
        if not isinstance(state, BoundedPIControllerState):
            raise TypeError("state must be a BoundedPIControllerState")
        setpoint_override = None
        if self.error_index is not None:
            observation = jnp.asarray(observation)
            measurement = observation[..., self.measurement_index]
            setpoint_override = measurement + observation[..., self.error_index]
        return step_from_observation(
            observation,
            state,
            kp=self.kp,
            ki=self.ki,
            ff=self.ff,
            output_min=self.output_min,
            output_max=self.output_max,
            dt=self.dt,
            anti_windup_gain=self.anti_windup_gain,
            max_ramp_up=self.max_ramp_up,
            max_ramp_down=self.max_ramp_down,
            measurement_index=self.measurement_index,
            setpoint_index=self.setpoint_index,
            setpoint_override=setpoint_override,
            feedforward_index=self.feedforward_index,
            feedforward_gain=self.feedforward_gain,
            bumpless=self.bumpless,
        )
