import jax.numpy as jnp
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
