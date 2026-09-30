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


def _hold_error(params: PIControllerParams, error: float, steps: int = 400):
    state = PIControllerState.create()
    for _ in range(steps):
        state, _ = pi_step(state, jnp.array(0.0), jnp.array(error), params, jnp.array(1.0))
    _, output = pi_step(state, jnp.array(0.0), jnp.array(0.0), params, jnp.array(1.0))
    return state, output


class TestDerivedIntegralLimit:
    @pytest.mark.parametrize(
        "ki, error, expected_bound",
        [(1.0, 0.1, 10.0), (1.0, -0.1, 0.0), (-0.5, -0.1, 10.0), (-0.5, 0.1, 0.0)],
        ids=["forward_high", "forward_low", "reverse_high", "reverse_low"],
    )
    def test_integral_alone_reaches_each_output_bound(self, ki: float, error: float, expected_bound: float) -> None:
        """Without an explicit clamp, a sustained error winds the integral far enough to hold either bound at zero error.

        The feed-forward sits off-centre so the two authorities differ (8 above,
        2 below). Conditional integration stops one increment short when the
        next increment would overshoot, hence the one-increment tolerance.
        """
        params = PIControllerParams(kp=0.0, ki=ki, ff=2.0, output_min=0.0, output_max=10.0)

        _, output = _hold_error(params, error)

        assert float(output) == pytest.approx(expected_bound, abs=abs(ki * error))

    def test_explicit_clamp_overrides_derived_one(self) -> None:
        """An explicit max_integral below the authority caps how far the integral can move the output."""
        params = PIControllerParams(kp=0.0, ki=1.0, ff=2.0, output_min=0.0, output_max=10.0, max_integral=1.0)

        _, output = _hold_error(params, 0.1)

        assert float(output) == pytest.approx(3.0)

    def test_zero_integral_gain_pins_integral_at_zero(self) -> None:
        """With no integral action there is no authority to size, so the integral never accumulates."""
        params = PIControllerParams(kp=0.0, ki=0.0, ff=2.0, output_min=0.0, output_max=10.0)

        state, output = _hold_error(params, 0.1)

        assert float(state.integral) == 0.0
        assert float(output) == pytest.approx(2.0)

    def test_tracking_is_bumpless_within_derived_clamp(self) -> None:
        """Manual-to-auto transfer stays bumpless when the clamp is derived rather than explicit."""
        params = PIControllerParams(kp=2.0, ki=0.5, ff=5.0, output_min=0.0, output_max=10.0)

        state = track_output(PIControllerState.create(), jnp.array(4.0), jnp.array(5.0), jnp.array(8.0), params)
        _, result = step_with_diagnostics(state, jnp.array(4.0), jnp.array(5.0), params, jnp.array(0.0))

        assert float(result.raw) == pytest.approx(8.0)

    def test_tracking_clips_integral_to_derived_clamp(self) -> None:
        """A tracked integral beyond the authority bound (5 / 0.5 = 10) is clipped to it."""
        params = PIControllerParams(kp=2.0, ki=0.5, ff=5.0, output_min=0.0, output_max=10.0)

        state = track_output(PIControllerState.create(), jnp.array(7.0), jnp.array(5.0), jnp.array(10.0), params)

        assert float(state.integral) == pytest.approx(10.0)
