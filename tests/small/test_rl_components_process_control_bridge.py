import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.dosing_system import SUPERVISORY
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_components.process_control_bridge import make_adapter, make_process_control_env


def test_chlorine_adapter_exposes_native_action_bounds() -> None:
    action_space = make_adapter("chlorine").action_space()

    assert action_space.shape == (1,)
    assert action_space.action_low is not None
    assert action_space.action_high is not None
    assert jnp.array_equal(action_space.action_low, jnp.array([0.0], dtype=jnp.float32))
    assert jnp.array_equal(action_space.action_high, jnp.array([5.0], dtype=jnp.float32))


def test_h2s_adapter_exposes_per_setpoint_action_bounds() -> None:
    """Expose physical pH, ORP, and level limits for the H2S action vector.

    Continuous-control consumers need these bounds to normalize each setpoint
    without collapsing the three physical ranges into one scalar interval.
    """
    action_space = make_adapter("h2s_scrubber").action_space()

    assert action_space.shape == (3,)
    assert action_space.action_low is not None
    assert action_space.action_high is not None
    assert jnp.array_equal(action_space.action_low, jnp.array([8.0, 400.0, 3.0], dtype=jnp.float32))
    assert jnp.array_equal(action_space.action_high, jnp.array([11.0, 800.0, 8.0], dtype=jnp.float32))


def test_h2s_env_tracks_configured_setpoint_bounds() -> None:
    """Reflect custom H2S setpoint limits in the canonical environment spec."""
    spec = make_process_control_env(
        "h2s_scrubber",
        ph_sp_min=8.5,
        ph_sp_max=10.5,
        orp_sp_min=450.0,
        orp_sp_max=750.0,
        level_sp_min=3.5,
        level_sp_max=7.5,
    ).spec()

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([8.5, 450.0, 3.5], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([10.5, 750.0, 7.5], dtype=jnp.float32))


def test_h2s_scenario_is_part_of_static_environment_identity() -> None:
    """Keep the selected H2S scenario visible to batch provenance."""
    assert make_process_control_env("h2s_scrubber", scenario="load_shift").spec().id.endswith(":load_shift")


def test_h2s_environment_supports_normalized_reset_and_step() -> None:
    """Allow TD3-style normalized actions to reach the H2S benchmark."""
    environment = make_action_normalization_wrapper(make_process_control_env("h2s_scrubber"))
    spec = environment.spec()
    reset = environment.reset(jax.random.PRNGKey(0))
    step = environment.step(jax.random.PRNGKey(1), reset.state, jnp.zeros(spec.action_shape))

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, -jnp.ones((3,), dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.ones((3,), dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(step.observation))
    assert jnp.isfinite(step.reward)


def test_h2s_normalized_extremes_reach_distinct_physical_controls() -> None:
    """Map normalized action extremes to distinct setpoints through the public bridge.

    The benchmark consumes physical setpoints, so double normalization would
    collapse both extremes near the lower end of each actuator range.
    """
    environment = make_action_normalization_wrapper(make_process_control_env("h2s_scrubber"))
    reset = environment.reset(jax.random.PRNGKey(0))
    low = environment.step(jax.random.PRNGKey(1), reset.state, -jnp.ones((3,)))
    high = environment.step(jax.random.PRNGKey(1), reset.state, jnp.ones((3,)))

    assert not jnp.allclose(low.observation, high.observation)
    assert not jnp.allclose(low.reward, high.reward)
    for name in ("caustic_pump", "bleach_pump", "makeup_pump"):
        assert not jnp.allclose(low.info[name], high.info[name])


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
