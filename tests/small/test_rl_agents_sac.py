"""Small tests for rl_agents.sac — config, network shapes, entropy math."""

from typing import Protocol, cast

import distrax
import jax
import jax.numpy as jnp
from rl_agents.sac import Actor, Critic, SACConfig
from rl_components.distributions import TanhNormalDiag


class _MutableSACConfig(Protocol):
    LEARNING_RATE: float


class TestSACConfig:
    def test_defaults(self) -> None:
        cfg = SACConfig()
        assert cfg.LEARNING_RATE == 3e-4
        assert cfg.BUFFER_SIZE == 100_000
        assert cfg.TAU == 0.005
        assert cfg.ALPHA == 0.2
        assert cfg.TARGET_ENTROPY is None

    def test_frozen(self) -> None:
        cfg = SACConfig()
        try:
            mutable_cfg = cast(_MutableSACConfig, cfg)
            mutable_cfg.LEARNING_RATE = 0.1
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_custom_config(self) -> None:
        cfg = SACConfig(TARGET_ENTROPY=-2.0, TAU=0.01)
        assert cfg.TARGET_ENTROPY == -2.0
        assert cfg.TAU == 0.01


class TestCritic:
    def test_output_shape(self) -> None:
        critic = Critic()
        obs = jnp.zeros((4,))
        action = jnp.zeros((2,))
        params = critic.init(jax.random.key(0), obs, action)
        q = critic.apply(params, obs, action)
        assert q.shape == ()

    def test_batch_output_shape(self) -> None:
        critic = Critic()
        obs = jnp.zeros((10, 4))
        action = jnp.zeros((10, 2))
        params = critic.init(jax.random.key(0), jnp.zeros((4,)), jnp.zeros((2,)))
        q = critic.apply(params, obs, action)
        assert q.shape == (10,)


class TestActor:
    def test_output_shape(self) -> None:
        actor = Actor(action_dim=2)
        obs = jnp.zeros((4,))
        params = actor.init(jax.random.key(0), obs)
        mean, log_std = cast(tuple[jax.Array, jax.Array], actor.apply(params, obs))
        assert mean.shape == (2,)
        assert log_std.shape == (2,)

    def test_log_std_clipped(self) -> None:
        actor = Actor(action_dim=1)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))
        _, log_std = cast(tuple[jax.Array, jax.Array], actor.apply(params, jnp.ones((4,)) * 1000))
        assert jnp.all(log_std <= 2.0)
        assert jnp.all(log_std >= -20.0)

    def test_sample_action_bounded(self) -> None:
        """Sampled actions should be in [-1, 1] due to tanh."""
        actor = Actor(action_dim=3)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))
        action, log_prob = actor.sample(params, jnp.ones((4,)), jax.random.key(1))
        assert action.shape == (3,)
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <= 1.0)
        assert log_prob.shape == ()

    def test_sample_returns_one_log_prob_per_observation(self) -> None:
        """SAC sampling reduces per-dimension log-probabilities per observation."""
        actor = Actor(action_dim=2)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))
        _, log_prob = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
            params, jnp.ones((3, 4)), jax.random.split(jax.random.key(1), 3)
        )
        assert log_prob.shape == (3,)

    def test_sample_log_prob_matches_previous_formula(self) -> None:
        """Actor sampling matches the old fixed-key tanh Gaussian formula."""
        actor = Actor(action_dim=2)
        observation = jnp.ones((4,))
        params = actor.init(jax.random.key(0), observation)
        key = jax.random.key(1)
        mean, log_std = cast(tuple[jax.Array, jax.Array], actor.apply(params, observation))
        raw = distrax.Normal(mean, jnp.exp(log_std)).sample(seed=key)
        action = jnp.tanh(raw)
        expected = jnp.sum(distrax.Normal(mean, jnp.exp(log_std)).log_prob(raw) - jnp.log(1.0 - action**2 + 1e-6))
        _, actual = actor.sample(params, observation, key)
        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_distribution_sample_log_prob_preserves_saturated_raw_value(self) -> None:
        """The helper uses the sampled pre-tanh value for saturated actions."""
        distribution = TanhNormalDiag(mean=jnp.array([12.0, -0.3]), log_std=jnp.full(2, -20.0))
        key = jax.random.key(0)
        raw = distrax.Normal(distribution.mean, jnp.exp(distribution.log_std)).sample(seed=key)
        expected_action = jnp.tanh(raw)
        expected_log_prob = distrax.Normal(distribution.mean, jnp.exp(distribution.log_std)).log_prob(raw)
        expected_log_prob -= jnp.log(1.0 - expected_action**2 + 1e-6)
        action, log_prob = distribution.sample_and_log_prob(key)
        assert jnp.allclose(action, expected_action)
        assert jnp.allclose(log_prob, expected_log_prob, atol=1e-6)


class TestSoftUpdate:
    def test_tau_one_is_hard_copy(self) -> None:
        """With tau=1.0, target should become identical to online."""
        online = {"w": jnp.array([1.0, 2.0])}
        target = {"w": jnp.array([0.0, 0.0])}
        tau = 1.0

        def soft_update(tp: jax.Array, p: jax.Array):
            return tau * p + (1.0 - tau) * tp

        updated = jax.tree_util.tree_map(soft_update, target, online)
        assert jnp.allclose(updated["w"], online["w"])

    def test_tau_zero_is_no_update(self) -> None:
        """With tau=0.0, target should stay the same."""
        online = {"w": jnp.array([1.0, 2.0])}
        target = {"w": jnp.array([3.0, 4.0])}
        tau = 0.0

        def soft_update(tp: jax.Array, p: jax.Array):
            return tau * p + (1.0 - tau) * tp

        updated = jax.tree_util.tree_map(soft_update, target, online)
        assert jnp.allclose(updated["w"], target["w"])
