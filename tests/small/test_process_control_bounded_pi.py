import jax
import jax.numpy as jnp
from process_control.controllers.bounded_pi import (
    BoundedPIControllerOutput,
    BoundedPIControllerParams,
    BoundedPIControllerState,
    step,
)


def test_bounded_pi_preserves_bumpless_initialization() -> None:
    """
    Initialize integral memory so the first automatic output equals the prior output.

    This protects manual-to-automatic transfer semantics.
    """
    params = BoundedPIControllerParams(
        kp=2.0,
        ki=0.5,
        ff=0.5,
        output_min=0.0,
        output_max=5.0,
        dt=1.0,
        max_ramp_up=0.25,
        max_ramp_down=0.25,
    )
    state, result = step(
        BoundedPIControllerState.create(2.0),
        jnp.array(0.0),
        jnp.array(1.0),
        params,
    )

    assert jnp.allclose(result.realized, jnp.array(2.0))
    assert jnp.allclose(state.previous_output, jnp.array(2.0))


def test_bounded_pi_applies_asymmetric_rate_limits() -> None:
    """
    Apply independent upward and downward actuator rates to PI requests.

    Hard output bounds remain active alongside the rate limits.
    """
    params = BoundedPIControllerParams(
        kp=4.0,
        ki=0.0,
        ff=0.0,
        output_min=0.0,
        output_max=1.0,
        dt=1.0,
        max_ramp_up=0.25,
        max_ramp_down=0.5,
    )
    state, first = step(
        BoundedPIControllerState.create(),
        jnp.array(0.0),
        jnp.array(1.0),
        params,
    )
    state, second = step(state, jnp.array(-1.0), jnp.array(1.0), params)
    _state, third = step(state, jnp.array(2.0), jnp.array(1.0), params)

    assert jnp.allclose(first.realized, jnp.array(0.25))
    assert jnp.allclose(second.realized, jnp.array(0.5))
    assert jnp.allclose(third.realized, jnp.array(0.0))


def test_bounded_pi_is_jittable_and_vmap_safe() -> None:
    """
    Preserve scalar state shape and values when transformed by JIT and vmap.

    Batched evaluation uses this same functional boundary.
    """
    params = BoundedPIControllerParams(
        kp=1.0,
        ki=0.25,
        ff=0.5,
        output_min=0.0,
        output_max=5.0,
        dt=0.5,
    )
    states = BoundedPIControllerState(
        integral_error=jnp.array([0.0, 0.5]),
        previous_output=jnp.array([2.0, 1.0]),
        initialized=jnp.array([True, True]),
    )
    measurements = jnp.array([0.25, 1.25])
    setpoints = jnp.array([1.5, 1.0])

    def step_one(
        item: BoundedPIControllerState,
        measurement: jax.Array,
        setpoint: jax.Array,
    ) -> tuple[BoundedPIControllerState, BoundedPIControllerOutput]:
        return step(item, measurement, setpoint, params)

    expected_state, expected = step(states, measurements, setpoints, params)
    actual_state, actual = jax.jit(jax.vmap(step_one))(
        states,
        measurements,
        setpoints,
    )

    assert jnp.allclose(actual.realized, expected.realized)
    assert jnp.allclose(actual_state.integral_error, expected_state.integral_error)
