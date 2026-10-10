import math

import jax.numpy as jnp
import pytest
from process_control.controllers.pi_controller import (
    BSM1_NITRATE_CONTROLLER,
    BSM1_OXYGEN_CONTROLLER,
    PIControllerParams,
    PIControllerState,
    standard_form,
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


class TestBackCalculation:
    PARAMS = PIControllerParams(kp=2.0, ki=4.0, ff=5.0, output_min=0.0, output_max=10.0, tracking_time=0.5)

    def _hold(self, params: PIControllerParams, error: float, steps: int, dt: float) -> tuple[PIControllerState, list[float]]:
        state = PIControllerState.create()
        raw: list[float] = []
        for _ in range(steps):
            state, result = step_with_diagnostics(state, jnp.array(0.0), jnp.array(error), params, jnp.array(dt))
            raw.append(float(result.raw))
        return state, raw

    def test_unsaturated_controller_integrates_the_error(self) -> None:
        """Inside the output range back-calculation is inactive and the integral accumulates e·dt."""
        state, raw = self._hold(self.PARAMS, error=0.1, steps=5, dt=0.1)

        assert float(state.integral) == pytest.approx(0.05)
        assert raw[-1] == pytest.approx(2.0 * 0.1 + 4.0 * 0.05 + 5.0)

    def test_sustained_saturation_settles_at_the_tracking_equilibrium(self) -> None:
        """Held in saturation, the raw output stops growing where the error input balances the tracking feedback.

        In continuous time that excess over the limit is ki·e·Tt.
        """
        state, raw = self._hold(self.PARAMS, error=1.0, steps=2000, dt=0.005)

        assert raw[-1] - 10.0 == pytest.approx(4.0 * 1.0 * 0.5, rel=0.01)
        assert raw[-1] == pytest.approx(raw[-2], abs=1e-6)

    def test_reversed_error_leaves_saturation_promptly(self) -> None:
        """After a long saturation, reversing the error brings the output off the limit within a tracking time."""
        state, _ = self._hold(self.PARAMS, error=5.0, steps=500, dt=0.01)

        recovered = []
        for _ in range(50):
            state, result = step_with_diagnostics(state, jnp.array(1.0), jnp.array(0.0), self.PARAMS, jnp.array(0.01))
            recovered.append(bool(result.is_saturated))

        assert not all(recovered)

    def test_steps_longer_than_the_tracking_time_stay_bounded(self) -> None:
        """With dt far above Tt the tracking term removes the excess without overshooting into oscillation."""
        params = PIControllerParams(kp=2.0, ki=4.0, ff=5.0, output_min=0.0, output_max=10.0, tracking_time=0.01)

        _, raw = self._hold(params, error=1.0, steps=50, dt=1.0)

        assert all(10.0 <= value <= 10.0 + 2.0 + 4.0 * 1.0 * 1.0 for value in raw[1:])

    def test_saturated_step_holds_the_tracking_equilibrium_exactly(self) -> None:
        """At the equilibrium of du/dt = ki·e + (u_sat − u)/Tt a whole finite step leaves the integral unchanged."""
        params = PIControllerParams(kp=1.0, ki=1.0, ff=0.0, output_min=0.0, output_max=10.0, tracking_time=1.0)

        state, result = step_with_diagnostics(PIControllerState(integral=jnp.array(10.0)), jnp.array(0.0), jnp.array(1.0), params, jnp.array(1.0))

        assert float(state.integral) == pytest.approx(10.0)
        assert float(result.raw) == pytest.approx(11.0)

    def test_step_leaving_saturation_follows_the_piecewise_solution(self) -> None:
        """A reversed error relaxes the raw output to the limit, then integrates linearly inside the range."""
        params = PIControllerParams(kp=1.0, ki=1.0, ff=0.0, output_min=0.0, output_max=10.0, tracking_time=1.0)
        start = PIControllerState(integral=jnp.array(20.0))

        state, result = step_with_diagnostics(start, jnp.array(3.0), jnp.array(0.0), params, jnp.array(2.0))

        # u relaxes from 17 toward 7 and reaches the limit 10 after ln(10/3), then falls at 3 per unit time.
        expected_raw = 10.0 - 3.0 * (2.0 - math.log(10.0 / 3.0))
        assert float(result.raw) == pytest.approx(expected_raw, rel=1e-5)
        assert float(state.integral) == pytest.approx(expected_raw + 3.0, rel=1e-5)

    def test_zero_integral_gain_keeps_the_integral(self) -> None:
        """A proportional-only controller has no integral to unwind."""
        params = PIControllerParams(kp=20.0, ki=0.0, ff=5.0, output_min=0.0, output_max=10.0, tracking_time=0.5)

        state, raw = self._hold(params, error=1.0, steps=3, dt=0.1)

        assert float(state.integral) == 0.0
        assert raw[-1] == pytest.approx(25.0)


class TestBSM1Presets:
    def test_standard_form_sets_the_integral_gain_from_the_integral_time(self) -> None:
        """K (e + (1/Ti) ∫e) has integral gain K / Ti."""
        params = standard_form(gain=3.0, integral_time=0.5, offset=1.0, output_min=0.0, output_max=10.0)

        assert params.ki == pytest.approx(6.0)
        assert params.tracking_time is None

    @pytest.mark.parametrize(
        ("params", "offset", "maximum"),
        [(BSM1_OXYGEN_CONTROLLER, 144.0 / 24.0, 360.0 / 24.0), (BSM1_NITRATE_CONTROLLER, 18500.0 / 24.0, 92230.0 / 24.0)],
    )
    def test_presets_output_their_offset_at_zero_error(self, params: PIControllerParams, offset: float, maximum: float) -> None:
        """At setpoint from rest the BSM1 controllers output their per-hour offsets within per-hour limits."""
        _, result = step_with_diagnostics(PIControllerState.create(), jnp.array(1.0), jnp.array(1.0), params, jnp.array(0.01))

        assert float(result.saturated) == pytest.approx(offset)
        assert params.output_max == pytest.approx(maximum)
