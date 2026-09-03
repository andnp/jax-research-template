import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.dosing_system import SUPERVISORY
from rl_components.env_protocol import EnvProtocol
from rl_components.process_control_bridge import make_adapter, make_process_control_env


def test_chlorine_adapter_exposes_native_action_bounds() -> None:
    action_space = make_adapter("chlorine").action_space()

    assert action_space.shape == (1,)
    assert action_space.action_low is not None
    assert action_space.action_high is not None
    assert jnp.array_equal(action_space.action_low, jnp.array([0.0], dtype=jnp.float32))
    assert jnp.array_equal(action_space.action_high, jnp.array([5.0], dtype=jnp.float32))


def test_unbounded_benchmark_preserves_missing_action_bounds() -> None:
    action_space = make_adapter("bsm1").action_space()

    assert action_space.shape == (2,)
    assert action_space.action_low is None
    assert action_space.action_high is None


def test_chlorine_adapter_tracks_configured_action_bounds() -> None:
    action_space = make_adapter(
        "chlorine",
        pump_min_dose=1.0,
        pump_max_dose=10.0,
    ).action_space()

    assert action_space.action_low is not None
    assert action_space.action_high is not None
    assert jnp.array_equal(action_space.action_low, jnp.array([1.0], dtype=jnp.float32))
    assert jnp.array_equal(action_space.action_high, jnp.array([10.0], dtype=jnp.float32))


def test_chlorine_non_direct_mode_omits_native_action_bounds() -> None:
    action_space = make_adapter("chlorine", control_mode=SUPERVISORY).action_space()

    assert action_space.action_low is None
    assert action_space.action_high is None


def test_process_control_env_conforms_to_env_protocol() -> None:
    assert isinstance(make_process_control_env("chlorine"), EnvProtocol)


def test_spec_reports_registry_shapes_and_native_action_bounds() -> None:
    spec = make_process_control_env("chlorine").spec()

    assert spec.id == "process_control:chlorine"
    assert spec.observation_shape == (4,)
    assert spec.action_shape == (1,)
    assert spec.observation_dtype == jnp.float32
    assert spec.action_dtype == jnp.float32
    assert spec.num_actions is None
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([0.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([5.0], dtype=jnp.float32))


def test_spec_preserves_missing_action_bounds_for_unbounded_benchmark() -> None:
    spec = make_process_control_env("bsm1").spec()

    assert spec.observation_shape == (9,)
    assert spec.action_shape == (2,)
    assert spec.action_low is None
    assert spec.action_high is None


def test_spec_tracks_configured_action_bounds() -> None:
    spec = make_process_control_env("chlorine", pump_min_dose=1.0, pump_max_dose=10.0).spec()

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([10.0], dtype=jnp.float32))


@pytest.mark.parametrize("benchmark_name", ["chlorine", "bsm1"])
def test_reset_and_step_return_identically_structured_pytrees(benchmark_name: str) -> None:
    env = make_process_control_env(benchmark_name)
    reset = env.reset(jax.random.PRNGKey(0))
    step = env.step(
        jax.random.PRNGKey(1),
        reset.state,
        jnp.zeros(env.spec().action_shape, dtype=jnp.float32),
    )

    reset_leaves, reset_structure = jax.tree.flatten((reset.observation, reset.state))
    step_leaves, step_structure = jax.tree.flatten((step.observation, step.state))

    assert reset_structure == step_structure
    assert [leaf.shape for leaf in reset_leaves] == [leaf.shape for leaf in step_leaves]
    assert [leaf.dtype for leaf in reset_leaves] == [leaf.dtype for leaf in step_leaves]


def test_step_maps_the_fused_done_onto_termination() -> None:
    env = make_process_control_env("chlorine")
    reset = env.reset(jax.random.PRNGKey(0))
    step = env.step(jax.random.PRNGKey(1), reset.state, jnp.array([0.5], dtype=jnp.float32))

    assert step.observation.shape == (4,)
    assert step.reward.shape == ()
    assert step.terminated.dtype == jnp.bool_
    assert step.truncated.dtype == jnp.bool_
    assert step.truncated.shape == step.terminated.shape
    assert not bool(jnp.any(step.truncated))
