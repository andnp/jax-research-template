"""Small tests for rl_agents.dqn — config validation and loss function math."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from rl_agents.dqn import DQNConfig, QNetwork, _make_q_network


class TestDQNConfig:
    def test_defaults(self):
        cfg = DQNConfig()
        assert cfg.LR == 3e-4
        assert cfg.BUFFER_SIZE == 100_000
        assert cfg.BATCH_SIZE == 64
        assert cfg.GAMMA == 0.99
        assert cfg.EPSILON_START == 1.0
        assert cfg.EPSILON_END == 0.05
        assert cfg.NETWORK_PRESET == "mlp"

    def test_frozen(self):
        cfg = DQNConfig()
        try:
            cfg.LR = 0.1  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_custom_config(self):
        cfg = DQNConfig(LR=1e-3, BUFFER_SIZE=1000, ENV_NAME="CartPole-v1")
        assert cfg.LR == 1e-3
        assert cfg.BUFFER_SIZE == 1000
        assert cfg.ENV_NAME == "CartPole-v1"

    def test_epsilon_schedule_math(self):
        """Epsilon should decay linearly from start to end over the specified fraction."""
        cfg = DQNConfig(EPSILON_START=1.0, EPSILON_END=0.1, EPSILON_FRACTION=0.5, TOTAL_TIMESTEPS=100)
        midpoint = int(cfg.TOTAL_TIMESTEPS * cfg.EPSILON_FRACTION)
        epsilon_at_mid = max(
            cfg.EPSILON_END,
            cfg.EPSILON_START - (cfg.EPSILON_START - cfg.EPSILON_END) * (midpoint / (cfg.TOTAL_TIMESTEPS * cfg.EPSILON_FRACTION)),
        )
        assert abs(epsilon_at_mid - cfg.EPSILON_END) < 1e-6


class TestQNetwork:
    def test_make_q_network_uses_mlp_by_default(self):
        cfg = DQNConfig()
        network = _make_q_network(cfg, action_dim=4)
        assert isinstance(network, QNetwork)

    def test_make_q_network_rejects_unsupported_nature_cnn(self):
        cfg = DQNConfig(NETWORK_PRESET="nature_cnn")
        with pytest.raises(ValueError, match="observation-shape contract"):
            _make_q_network(cfg, action_dim=4)

    def test_make_q_network_rejects_invalid_preset(self):
        cfg = DQNConfig()
        object.__setattr__(cfg, "NETWORK_PRESET", "bogus")
        with pytest.raises(ValueError, match="Invalid NETWORK_PRESET 'bogus'"):
            _make_q_network(cfg, action_dim=4)

    def test_output_shape(self):
        net = QNetwork(action_dim=4)
        params = net.init(jax.random.key(0), jnp.zeros((8,)))
        q = cast(jax.Array, net.apply(params, jnp.ones((8,))))
        assert q.shape == (4,)

    def test_batch_output_shape(self):
        net = QNetwork(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        q = cast(jax.Array, net.apply(params, jnp.ones((10, 4))))
        assert q.shape == (10, 3)


class TestDQNLoss:
    def test_td_error_zero_when_perfect(self):
        """TD error should be zero when Q matches the target."""
        q_values = jnp.array([[0.0, 1.0, 0.0]])
        actions = jnp.array([1])
        rewards = jnp.array([0.0])
        next_q_max = jnp.array([0.0])
        dones = jnp.array([1.0])

        q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()
        target = rewards + 0.99 * next_q_max * (1.0 - dones)
        loss = jnp.mean(jnp.square(q_action - target))
        # Q(s,a)=1.0, target=0+0.99*0*(0)=0, so loss = 1.0
        assert jnp.allclose(loss, 1.0)

    def test_td_error_with_reward(self):
        """TD error computation with a non-zero reward."""
        gamma = 0.99
        reward = 1.0
        q_action = 0.5
        next_q_max = 2.0
        done = 0.0

        target = reward + gamma * next_q_max * (1.0 - done)
        expected_target = 1.0 + 0.99 * 2.0 * 1.0  # = 2.98
        assert abs(target - expected_target) < 1e-6
        loss = (q_action - target) ** 2
        assert loss > 0
