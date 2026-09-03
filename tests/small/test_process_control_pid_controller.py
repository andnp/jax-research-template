import jax.numpy as jnp
import pytest
from process_control.controllers.pid_controller import (
    PIDControllerParams,
    PIDControllerState,
    step_with_diagnostics,
)


def test_derivative_term_reacts_to_measurement_rise() -> None:
    params = PIDControllerParams(
        kp=0.0,
        ki=0.0,
        kd=2.0,
        ff=5.0,
        output_min=0.0,
        output_max=10.0,
        max_integral=100.0,
    )
    state = PIDControllerState.create()
    state, first = step_with_diagnostics(
        state,
        jnp.array(4.0),
        jnp.array(5.0),
        params,
        jnp.array(1.0),
    )
    _, second = step_with_diagnostics(
        state,
        jnp.array(5.0),
        jnp.array(5.0),
        params,
        jnp.array(1.0),
    )

    assert float(first.saturated) == pytest.approx(5.0)
    assert float(second.saturated) == pytest.approx(3.0)


def test_pid_preserves_pi_behavior_when_derivative_is_zero() -> None:
    params = PIDControllerParams(
        kp=2.0,
        ki=1.0,
        kd=0.0,
        ff=5.0,
        output_min=0.0,
        output_max=10.0,
        max_integral=100.0,
    )
    state, result = step_with_diagnostics(
        PIDControllerState.create(),
        jnp.array(4.0),
        jnp.array(5.0),
        params,
        jnp.array(1.0),
    )

    assert float(result.saturated) == pytest.approx(8.0)
    assert bool(state.initialized)
