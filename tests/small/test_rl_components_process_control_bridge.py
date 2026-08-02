import jax.numpy as jnp
from process_control.actuators.dosing_system import SUPERVISORY
from rl_components.process_control_bridge import make_adapter


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
