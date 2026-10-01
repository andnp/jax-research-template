"""Small tests for rl_agents.dqn — config validation and network selection."""

import jax
import jax.numpy as jnp
import pytest
from rl_agents.dqn import DQNConfig
from rl_networks.q_networks import NatureQNetwork, QNetwork, make_q_network


class TestDQNConfig:
    def test_defaults(self) -> None:
        cfg = DQNConfig()
        assert cfg.learning_rate == 3e-4
        assert cfg.buffer_size == 100_000
        assert cfg.batch_size == 64
        assert cfg.epsilon_start == 1.0
        assert cfg.epsilon_end == 0.05
        assert cfg.network_preset == "mlp"

    def test_custom_config(self) -> None:
        cfg = DQNConfig(learning_rate=1e-3, buffer_size=1000, env_name="CartPole-v1")
        assert cfg.learning_rate == 1e-3
        assert cfg.buffer_size == 1000
        assert cfg.env_name == "CartPole-v1"


class TestQNetwork:
    def test_make_q_network_uses_mlp_by_default(self) -> None:
        cfg = DQNConfig()
        network = make_q_network(cfg, action_dim=4)
        assert isinstance(network, QNetwork)

    def test_make_q_network_uses_nature_cnn_for_atari_style_observations(self) -> None:
        cfg = DQNConfig(network_preset="nature_cnn")
        network = make_q_network(cfg, action_dim=4, observation_shape=(4, 84, 84, 1))
        assert isinstance(network, NatureQNetwork)

    def test_make_q_network_requires_observation_shape_for_nature_cnn(self) -> None:
        cfg = DQNConfig(network_preset="nature_cnn")
        with pytest.raises(ValueError, match="requires observation_shape"):
            make_q_network(cfg, action_dim=4)

    def test_make_q_network_rejects_non_image_observation_shape_for_nature_cnn(self) -> None:
        cfg = DQNConfig(network_preset="nature_cnn")
        with pytest.raises(ValueError, match="requires image observations"):
            make_q_network(cfg, action_dim=4, observation_shape=(8,))

    def test_make_q_network_rejects_invalid_preset(self) -> None:
        cfg = DQNConfig()
        object.__setattr__(cfg, "network_preset", "bogus")
        with pytest.raises(ValueError, match="Invalid network_preset 'bogus'"):
            make_q_network(cfg, action_dim=4)

    def test_output_shape(self) -> None:
        net = QNetwork(action_dim=4)
        params = net.init(jax.random.key(0), jnp.zeros((8,)))
        q = net.apply(params, jnp.ones((8,)))
        assert q.shape == (4,)

    def test_batch_output_shape(self) -> None:
        net = QNetwork(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        q = net.apply(params, jnp.ones((10, 4)))
        assert q.shape == (10, 3)

    def test_nature_q_network_output_shape_for_atari_observation(self) -> None:
        net = make_q_network(DQNConfig(network_preset="nature_cnn"), action_dim=3, observation_shape=(4, 84, 84, 1))
        x = jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8)
        params = net.init(jax.random.key(0), x)
        q = net.apply(params, x)
        assert q.shape == (3,)

    def test_nature_q_network_output_shape_for_batched_atari_observation(self) -> None:
        net = make_q_network(DQNConfig(network_preset="nature_cnn"), action_dim=3, observation_shape=(4, 84, 84, 1))
        x = jnp.zeros((2, 4, 84, 84, 1), dtype=jnp.uint8)
        params = net.init(jax.random.key(0), jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8))
        q = net.apply(params, x)
        assert q.shape == (2, 3)

    def test_nature_q_network_matches_pre_stacked_channel_layout(self) -> None:
        key = jax.random.key(0)
        frame_stacked = jnp.arange(4 * 84 * 84, dtype=jnp.uint8).reshape(4, 84, 84, 1)
        channel_stacked = jnp.moveaxis(frame_stacked, 0, -2).reshape(84, 84, 4)

        atari_net = NatureQNetwork(action_dim=3, observation_layout="fhwc")
        channel_last_net = NatureQNetwork(action_dim=3, observation_layout="hwc")

        atari_params = atari_net.init(key, frame_stacked)
        channel_last_params = channel_last_net.init(key, channel_stacked)
        atari_q = atari_net.apply(atari_params, frame_stacked)
        channel_last_q = channel_last_net.apply(channel_last_params, channel_stacked)

        assert jnp.allclose(atari_q, channel_last_q)
