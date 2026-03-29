"""Small tests for rl_agents.sac — config, network shapes, entropy math."""

from typing import Protocol, cast

import jax
import jax.numpy as jnp
from rl_agents.sac import Actor, Critic, SACConfig


class _MutableSACConfig(Protocol):
    LR: float


class TestSACConfig:
    def test_defaults(self):
        cfg = SACConfig()
        assert cfg.LR == 3e-4
        assert cfg.BUFFER_SIZE == 100_000
        assert cfg.GAMMA == 0.99
        assert cfg.TAU == 0.005
        assert cfg.ALPHA == 0.2
        assert cfg.TARGET_ENTROPY is None

    def test_frozen(self):
        cfg = SACConfig()
        try:
            mutable_cfg = cast(_MutableSACConfig, cfg)
            mutable_cfg.LR = 0.1
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_custom_config(self):
        cfg = SACConfig(TARGET_ENTROPY=-2.0, TAU=0.01)
        assert cfg.TARGET_ENTROPY == -2.0
        assert cfg.TAU == 0.01


class TestCritic:
    def test_output_shape(self):
        critic = Critic()
        obs = jnp.zeros((4,))
        action = jnp.zeros((2,))
        params = critic.init(jax.random.key(0), obs, action)
        q = cast(jax.Array, critic.apply(params, obs, action))
        assert q.shape == ()

    def test_batch_output_shape(self):
        critic = Critic()
        obs = jnp.zeros((10, 4))
        action = jnp.zeros((10, 2))
        params = critic.init(jax.random.key(0), jnp.zeros((4,)), jnp.zeros((2,)))
        q = cast(jax.Array, critic.apply(params, obs, action))
        assert q.shape == (10,)


class TestActor:
    def test_output_shape(self):
        actor = Actor(action_dim=2)
        obs = jnp.zeros((4,))
        params = actor.init(jax.random.key(0), obs)
        mean, log_std = cast(tuple[jax.Array, jax.Array], actor.apply(params, obs))
        assert mean.shape == (2,)
        assert log_std.shape == (2,)

    def test_log_std_clipped(self):
        actor = Actor(action_dim=1)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))
        _, log_std = cast(tuple[jax.Array, jax.Array], actor.apply(params, jnp.ones((4,)) * 1000))
        assert jnp.all(log_std <= 2.0)
        assert jnp.all(log_std >= -20.0)

    def test_sample_action_bounded(self):
        """Sampled actions should be in [-1, 1] due to tanh."""
        actor = Actor(action_dim=3)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))
        action, log_prob = actor.sample(params, jnp.ones((4,)), jax.random.key(1))
        assert action.shape == (3,)
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <= 1.0)
        assert log_prob.shape == ()


class TestSoftUpdate:
    def test_tau_one_is_hard_copy(self):
        """With tau=1.0, target should become identical to online."""
        online = {"w": jnp.array([1.0, 2.0])}
        target = {"w": jnp.array([0.0, 0.0])}
        tau = 1.0
        updated = jax.tree_util.tree_map(
            lambda tp, p: tau * p + (1.0 - tau) * tp, target, online
        )
        assert jnp.allclose(updated["w"], online["w"])

    def test_tau_zero_is_no_update(self):
        """With tau=0.0, target should stay the same."""
        online = {"w": jnp.array([1.0, 2.0])}
        target = {"w": jnp.array([3.0, 4.0])}
        tau = 0.0
        updated = jax.tree_util.tree_map(
            lambda tp, p: tau * p + (1.0 - tau) * tp, target, online
        )
        assert jnp.allclose(updated["w"], target["w"])
