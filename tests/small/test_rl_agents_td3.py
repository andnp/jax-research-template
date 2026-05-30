"""Small tests for rl_agents.td3 — config, network shapes, soft update."""

from typing import Protocol, cast

import jax
import jax.numpy as jnp
from rl_agents.td3 import Actor, Critic, TD3Config


class _MutableTD3Config(Protocol):
    LR: float


class TestTD3Config:
    def test_defaults(self) -> None:
        cfg = TD3Config()
        assert cfg.LR == 3e-4
        assert cfg.BUFFER_SIZE == 100_000
        assert cfg.BATCH_SIZE == 256
        assert cfg.TOTAL_TIMESTEPS == 1_000_000
        assert cfg.LEARNING_STARTS == 25_000
        assert cfg.TRAIN_FREQUENCY == 1
        assert cfg.GAMMA == 0.99
        assert cfg.TAU == 0.005
        assert cfg.POLICY_DELAY == 2
        assert cfg.EXPLORATION_NOISE == 0.1
        assert cfg.POLICY_NOISE == 0.2
        assert cfg.NOISE_CLIP == 0.5
        assert cfg.ENV_NAME == "MountainCarContinuous-v0"
        assert cfg.SEED == 42

    def test_frozen(self) -> None:
        cfg = TD3Config()
        try:
            mutable_cfg = cast(_MutableTD3Config, cfg)
            mutable_cfg.LR = 0.1
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_custom_config(self) -> None:
        cfg = TD3Config(POLICY_DELAY=3, TAU=0.01, EXPLORATION_NOISE=0.2)
        assert cfg.POLICY_DELAY == 3
        assert cfg.TAU == 0.01
        assert cfg.EXPLORATION_NOISE == 0.2


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
