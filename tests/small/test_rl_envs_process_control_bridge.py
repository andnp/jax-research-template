import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.dosing_system import SUPERVISORY
from process_control.benchmarks.bsm1 import bsm1_recycle_control
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_envs.process_control_bridge import make_process_control_env


def test_chlorine_env_exposes_native_action_bounds() -> None:
    spec = make_process_control_env("chlorine").spec()

    assert spec.action_shape == (1,)
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([0.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([8.0], dtype=jnp.float32))


def test_h2s_env_exposes_per_setpoint_action_bounds() -> None:
    """Expose physical pH, ORP, and flow limits for the H2S action vector.

    Continuous-control consumers need these bounds to normalize each setpoint
    without collapsing the three physical ranges into one scalar interval.
    """
    spec = make_process_control_env("h2s_scrubber").spec()

    assert spec.action_shape == (3,)
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([8.0, 400.0, 10.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([9.8, 800.0, 30.0], dtype=jnp.float32))


def test_h2s_env_tracks_configured_setpoint_bounds() -> None:
    """Reflect custom H2S setpoint limits in the canonical environment spec."""
    spec = make_process_control_env(
        "h2s_scrubber",
        ph_sp_min=8.5,
        ph_sp_max=10.5,
        orp_sp_min=450.0,
        orp_sp_max=750.0,
        makeup_flow_sp_min=12.0,
        makeup_flow_sp_max=28.0,
    ).spec()

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([8.5, 450.0, 12.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([10.5, 750.0, 28.0], dtype=jnp.float32))


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
    assert any(
        not jnp.allclose(low.info[name], high.info[name])
        for name in ("caustic_pump", "bleach_pump", "makeup_pump")
    )


def test_bsm1_exposes_its_configured_action_space() -> None:
    """The BSM1 action layout follows its handle configuration, in the Table 15 limits per hour."""
    with jax.enable_x64(True):
        default = make_process_control_env("bsm1").spec()
        recycle = make_process_control_env("bsm1", handles=bsm1_recycle_control().handles).spec()

    assert default.action_shape == (3,)
    assert default.action_low is not None
    assert default.action_high is not None
    assert jnp.array_equal(default.action_low, jnp.zeros(3, dtype=jnp.float32))
    assert jnp.allclose(default.action_high, jnp.array([15.0, 15.0, 1844.6 / 24.0], dtype=jnp.float32))
    assert recycle.action_shape == (2,)


def test_bsm1_needs_float64() -> None:
    """Building BSM1 without x64 fails loudly instead of stalling its slow states in float32."""
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="float64"):
        make_process_control_env("bsm1")


def test_chlorine_env_tracks_configured_action_bounds() -> None:
    spec = make_process_control_env(
        "chlorine",
        pump_min_dose=1.0,
        pump_max_dose=10.0,
    ).spec()

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([10.0], dtype=jnp.float32))


def test_equalization_tank_exposes_native_action_bounds() -> None:
    spec = make_process_control_env("equalization_tank").spec()

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([0.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([150.0], dtype=jnp.float32))


def test_ph_neutralization_exposes_one_native_bound_per_pump_stage() -> None:
    spec = make_process_control_env("ph_neutralization").spec()

    assert spec.action_shape == (2,)
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([0.0, 0.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([2.1, 0.9], dtype=jnp.float32))


def test_chlorine_non_direct_mode_omits_native_action_bounds() -> None:
    spec = make_process_control_env("chlorine", control_mode=SUPERVISORY).spec()

    assert spec.action_low is None
    assert spec.action_high is None


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
    assert jnp.array_equal(spec.action_high, jnp.array([8.0], dtype=jnp.float32))


def test_spec_tracks_configured_action_bounds() -> None:
    spec = make_process_control_env("chlorine", pump_min_dose=1.0, pump_max_dose=10.0).spec()

    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([10.0], dtype=jnp.float32))


@pytest.mark.parametrize(("benchmark_name", "x64"), [("chlorine", False), ("bsm1", True)])
def test_reset_and_step_return_identically_structured_pytrees(benchmark_name: str, x64: bool) -> None:
    with jax.enable_x64(x64):
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
