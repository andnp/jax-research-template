import jax
import jax.numpy as jnp
import pytest
from process_control.controllers.bounded_pi import (
    BoundedPIController,
    BoundedPIControllerParams,
    BoundedPIControllerState,
    step_from_observation,
)


def _observation(measured: float, setpoint: float = 1.0) -> jax.Array:
    return jnp.array([0.0, measured, setpoint, 10.0])


def test_bounded_pi_adapter_preserves_bumpless_initial_output() -> None:
    """The first action preserves the configured automatic-transfer output."""
    controller = BoundedPIController(
        kp=2.0,
        ki=0.5,
        ff=0.5,
        output_min=0.0,
        output_max=5.0,
        initial_output=2.0,
    )

    state = controller.reset(jax.random.PRNGKey(0))
    action, updated_state = controller.act(
        _observation(measured=0.0),
        state,
        jax.random.PRNGKey(1),
    )

    assert action.shape == (1,)
    assert jnp.allclose(action, jnp.array([2.0]))
    assert bool(updated_state.initialized)


def test_bounded_pi_adapter_maps_observation_fields() -> None:
    """Measurement, feedforward, and derived-error fields reach the PI step."""
    controller = BoundedPIController(
        kp=2.0,
        ki=0.0,
        ff=1.0,
        output_min=0.0,
        output_max=30.0,
        measurement_index=0,
        setpoint_index=0,
        error_index=2,
        feedforward_index=3,
        feedforward_gain=3.0,
    )
    state = controller.reset(jax.random.PRNGKey(0))

    action, _ = controller.act(
        jnp.array([1.0, 0.0, 0.5, 5.0]),
        state,
        jax.random.PRNGKey(1),
    )

    assert jnp.allclose(action, jnp.array([17.0]))


def test_bounded_pi_adapter_matches_functional_transition() -> None:
    """The stateful adapter preserves the canonical functional transition."""
    controller = BoundedPIController(
        kp=1.5,
        ki=0.25,
        ff=0.5,
        output_min=0.0,
        output_max=5.0,
        dt=0.5,
        anti_windup_gain=0.75,
        max_ramp_up=0.8,
        max_ramp_down=0.6,
        feedforward_index=3,
        feedforward_gain=0.1,
        initial_output=2.0,
    )
    observation = _observation(measured=0.25, setpoint=1.5)
    state = controller.reset(jax.random.PRNGKey(0))
    params = BoundedPIControllerParams(
        kp=controller.kp,
        ki=controller.ki,
        ff=controller.ff,
        output_min=controller.output_min,
        output_max=controller.output_max,
        dt=controller.dt,
        anti_windup_gain=controller.anti_windup_gain,
        max_ramp_up=controller.max_ramp_up,
        max_ramp_down=controller.max_ramp_down,
    )

    expected_action, expected_state = step_from_observation(
        observation,
        state,
        kp=params.kp,
        ki=params.ki,
        ff=params.ff,
        output_min=params.output_min,
        output_max=params.output_max,
        dt=params.dt,
        anti_windup_gain=params.anti_windup_gain,
        max_ramp_up=params.max_ramp_up,
        max_ramp_down=params.max_ramp_down,
        feedforward_index=3,
        feedforward_gain=0.1,
    )
    actual_action, actual_state = controller.act(
        observation,
        state,
        jax.random.PRNGKey(1),
    )

    assert jnp.allclose(actual_action, expected_action)
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(actual_state),
            jax.tree_util.tree_leaves(expected_state),
            strict=True,
        )
    )


def test_bounded_pi_adapter_supports_jit_and_vmap() -> None:
    """Batched adapter calls retain action shape and transformed state updates."""
    controller = BoundedPIController(
        kp=1.0,
        ki=0.2,
        ff=0.5,
        output_min=0.0,
        output_max=5.0,
        dt=0.5,
        initial_output=1.5,
    )
    observations = jnp.array(
        [[0.0, 0.25, 1.5, 10.0], [0.0, 1.25, 1.0, 20.0]],
        dtype=jnp.float32,
    )
    state = BoundedPIControllerState(
        integral_error=jnp.zeros(2, dtype=jnp.float32),
        previous_output=jnp.array([1.5, 1.5], dtype=jnp.float32),
        initialized=jnp.array([False, False]),
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    actions, updated_state = jax.jit(jax.vmap(controller.act))(
        observations,
        state,
        keys,
    )

    assert actions.shape == (2, 1)
    assert actions.dtype == observations.dtype
    assert jnp.allclose(actions, state.previous_output[:, None])
    assert jnp.all(updated_state.initialized)


def test_bounded_pi_adapter_rejects_invalid_state() -> None:
    """The adapter rejects state values outside its typed runtime contract."""
    controller = BoundedPIController(kp=1.0, ki=0.2)

    with pytest.raises(TypeError, match="BoundedPIControllerState"):
        controller.act(_observation(measured=0.0), object(), jax.random.PRNGKey(0))


def test_bounded_pi_controller_validates_configuration() -> None:
    """Invalid limits and timing settings fail before controller execution."""
    with pytest.raises(ValueError, match="output_max"):
        BoundedPIController(kp=1.0, ki=0.2, output_min=1.0, output_max=1.0)
    with pytest.raises(ValueError, match="dt"):
        BoundedPIController(kp=1.0, ki=0.2, dt=0.0)
    with pytest.raises(ValueError, match="anti_windup_gain"):
        BoundedPIController(kp=1.0, ki=0.2, anti_windup_gain=-1.0)
    with pytest.raises(ValueError, match="max_ramp_up"):
        BoundedPIController(kp=1.0, ki=0.2, max_ramp_up=0.0)
    with pytest.raises(ValueError, match="max_ramp_down"):
        BoundedPIController(kp=1.0, ki=0.2, max_ramp_down=0.0)


def test_step_from_observation_preserves_core_state_type() -> None:
    """Observation adaptation returns the same JAX-native state contract."""
    state = BoundedPIControllerState.create(1.0)

    _action, updated_state = step_from_observation(
        _observation(measured=0.0),
        state,
        kp=1.0,
        ki=0.2,
        ff=0.0,
        output_min=0.0,
        output_max=2.0,
        dt=1.0,
    )

    assert isinstance(updated_state, BoundedPIControllerState)
