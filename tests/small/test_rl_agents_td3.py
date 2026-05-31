"""Small tests for rl_agents.td3 — network shapes, soft update."""

from typing import cast

import jax
import jax.numpy as jnp
from rl_agents.td3 import Actor, Critic


class TestCritic:
    def test_output_shape(self) -> None:
        critic = Critic()
        obs = jnp.zeros((4,))
        action = jnp.zeros((2,))
        params = critic.init(jax.random.key(0), obs, action)
        q = cast(jax.Array, critic.apply(params, obs, action))
        assert q.shape == ()

    def test_batch_output_shape(self) -> None:
        critic = Critic()
        obs = jnp.zeros((10, 4))
        action = jnp.zeros((10, 2))
        params = critic.init(jax.random.key(0), jnp.zeros((4,)), jnp.zeros((2,)))
        q = cast(jax.Array, critic.apply(params, obs, action))
        assert q.shape == (10,)


class TestActor:
    def test_output_shape(self) -> None:
        actor = Actor(action_dim=2)
        obs = jnp.zeros((4,))
        params = actor.init(jax.random.key(0), obs)
        action = cast(jax.Array, actor.apply(params, obs))
        assert action.shape == (2,)

    def test_action_bounded(self) -> None:
        actor = Actor(action_dim=3)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))
        action = cast(jax.Array, actor.apply(params, jnp.ones((4,)) * 100))
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <= 1.0)


class TestSoftUpdate:
    def test_tau_one_is_hard_copy(self) -> None:
        online = {"w": jnp.array([1.0, 2.0])}
        target = {"w": jnp.array([0.0, 0.0])}
        tau = 1.0
        updated = jax.tree_util.tree_map(
            lambda tp, p: tau * p + (1.0 - tau) * tp, target, online
        )
        assert jnp.allclose(updated["w"], online["w"])

    def test_tau_zero_is_no_update(self) -> None:
        online = {"w": jnp.array([1.0, 2.0])}
        target = {"w": jnp.array([3.0, 4.0])}
        tau = 0.0
        updated = jax.tree_util.tree_map(
            lambda tp, p: tau * p + (1.0 - tau) * tp, target, online
        )
        assert jnp.allclose(updated["w"], target["w"])
