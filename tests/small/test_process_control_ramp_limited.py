import jax.numpy as jnp
from process_control.actuators.ramp_limited import (
    RampLimitedActuatorParams,
    RampLimitedActuatorState,
    step_with_diagnostics,
)


def test_ramp_limited_actuator_reports_saturation_and_rate_activation() -> None:
    """
    Distinguish hard output saturation from rate limiting in actuator telemetry.

    Both diagnostics are needed for benchmark-independent evaluation metrics.
    """
    params = RampLimitedActuatorParams(
        max_output=1.0,
        min_output=0.0,
        max_ramp_up=0.25,
        max_ramp_down=0.5,
    )
    state, result = step_with_diagnostics(
        RampLimitedActuatorState.create(),
        jnp.array(4.0),
        params,
        jnp.array(1.0),
    )

    assert jnp.allclose(state.current_output, jnp.array(0.25))
    assert bool(result.is_saturated)
    assert bool(result.is_rate_limited)


def test_symmetric_rate_rate_remains_backward_compatible() -> None:
    """
    Treat the legacy symmetric ramp parameter as both directional limits.

    Existing actuator callers retain their trajectory semantics.
    """
    params = RampLimitedActuatorParams(max_output=5.0, min_output=0.0, max_ramp_rate=1.0)
    _state, result = step_with_diagnostics(
        RampLimitedActuatorState.create(),
        jnp.array(5.0),
        params,
        jnp.array(1.0),
    )

    assert jnp.allclose(result.realized, jnp.array(1.0))
