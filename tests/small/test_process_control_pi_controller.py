import jax.numpy as jnp
import pytest
from process_control.controllers.pi_controller import (
    PIControllerParams,
    PIControllerState,
    step_with_diagnostics,
    track_output,
)
from process_control.controllers.pi_controller import step as pi_step


@pytest.fixture
def params() -> PIControllerParams:
    return PIControllerParams(
        kp=2.0,
        ki=1.0,
        ff=5.0,
        output_min=0.0,
        output_max=10.0,
        max_integral=100.0,
    )


def test_high_saturation_does_not_wind_up(params: PIControllerParams) -> None:
    state = PIControllerState.create()

    for _ in range(100):
        state, result = step_with_diagnostics(state, jnp.array(0.0), jnp.array(10.0), params, jnp.array(1.0))

    assert float(state.integral) == pytest.approx(0.0)
    assert float(result.raw) > params.output_max
    assert float(result.saturated) == params.output_max
    assert bool(result.is_saturated)


def test_low_saturation_does_not_wind_down(params: PIControllerParams) -> None:
    state = PIControllerState.create()

    for _ in range(100):
        state, result = step_with_diagnostics(state, jnp.array(10.0), jnp.array(0.0), params, jnp.array(1.0))

    assert float(state.integral) == pytest.approx(0.0)
    assert float(result.raw) < params.output_min
    assert float(result.saturated) == params.output_min
    assert bool(result.is_saturated)


def test_reverse_acting_controller_does_not_wind_up() -> None:
    params = PIControllerParams(-2.0, -1.0, 5.0, 0.0, 10.0, 100.0)
    state = PIControllerState.create()

    for _ in range(100):
        state, result = step_with_diagnostics(state, jnp.array(10.0), jnp.array(0.0), params, jnp.array(1.0))

    assert float(state.integral) == pytest.approx(0.0)
    assert float(result.saturated) == params.output_max


def test_controller_recovers_immediately_after_saturation(params: PIControllerParams) -> None:
    state = PIControllerState.create()
    for _ in range(100):
        state, _ = pi_step(state, jnp.array(0.0), jnp.array(10.0), params, jnp.array(1.0))

    state, output = pi_step(state, jnp.array(6.0), jnp.array(5.0), params, jnp.array(1.0))

    assert float(state.integral) == pytest.approx(-1.0)
    assert float(output) == pytest.approx(2.0)


def test_tracking_makes_manual_to_auto_transfer_bumpless(params: PIControllerParams) -> None:
    state = track_output(
        PIControllerState.create(),
        measurement=jnp.array(4.0),
        setpoint=jnp.array(5.0),
        output=jnp.array(8.0),
        params=params,
    )

    _, result = step_with_diagnostics(
        state,
        measurement=jnp.array(4.0),
        setpoint=jnp.array(5.0),
        params=params,
        dt=jnp.array(0.0),
    )

    assert float(result.raw) == pytest.approx(8.0)
    assert float(result.saturated) == pytest.approx(8.0)


def test_tracking_with_zero_integral_gain_preserves_state() -> None:
    params = PIControllerParams(2.0, 0.0, 5.0, 0.0, 10.0, 100.0)
    state = PIControllerState(integral=jnp.array(3.0))

    tracked = track_output(state, jnp.array(4.0), jnp.array(5.0), jnp.array(8.0), params)

    assert float(tracked.integral) == pytest.approx(3.0)
