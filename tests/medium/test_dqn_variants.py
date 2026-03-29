"""Medium tests for DQN variants — Double DQN and Dueling DQN gradient flow."""

from typing import cast

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from rl_agents.double_dqn import DoubleDQNConfig
from rl_agents.dqn import NatureQNetwork, _make_q_network
from rl_agents.dueling_dqn import DuelingDQNConfig, DuelingQNetwork, _make_dueling_q_network


class TestDoubleDQNGradient:
    def test_double_dqn_target_uses_online_for_selection(self):
        """In Double DQN, online net selects actions, target net evaluates them."""
        net = _make_q_network(DoubleDQNConfig(), action_dim=2, observation_shape=(4,))
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))
        target_params = net.init(jax.random.key(99), jnp.zeros((4,)))

        obs = jnp.ones((8, 4))
        # Online net selects best actions
        next_q_online = cast(jax.Array, net.apply(params, obs))
        next_actions = jnp.argmax(next_q_online, axis=-1)
        # Target net evaluates them
        next_q_target = cast(jax.Array, net.apply(target_params, obs))
        next_q_value = jnp.take_along_axis(next_q_target, next_actions[:, None], axis=-1).squeeze()
        assert next_q_value.shape == (8,)

    def test_double_dqn_params_change(self):
        net = _make_q_network(DoubleDQNConfig(), action_dim=2, observation_shape=(4,))
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        target_params = params

        tx = optax.adam(3e-4)
        state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        obs = jax.random.normal(jax.random.key(1), (16, 4))
        actions = jax.random.randint(jax.random.key(2), (16,), 0, 2)
        rewards = jax.random.normal(jax.random.key(3), (16,))
        next_obs = jax.random.normal(jax.random.key(4), (16, 4))
        dones = jnp.zeros((16,))

        def loss_fn(params):
            q_values = cast(jax.Array, net.apply(params, obs))
            q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()
            next_q_online = cast(jax.Array, net.apply(params, next_obs))
            next_actions = jnp.argmax(next_q_online, axis=-1)
            next_q_target = cast(jax.Array, net.apply(target_params, next_obs))
            next_q_value = jnp.take_along_axis(next_q_target, next_actions[:, None], axis=-1).squeeze()
            target = rewards + 0.99 * next_q_value * (1.0 - dones)
            return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))

    def test_double_dqn_can_use_nature_preset(self):
        net = _make_q_network(
            DoubleDQNConfig(NETWORK_PRESET="nature_cnn"),
            action_dim=3,
            observation_shape=(4, 84, 84, 1),
        )
        assert isinstance(net, NatureQNetwork)
        params = net.init(jax.random.key(0), jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8))
        q = cast(jax.Array, net.apply(params, jnp.zeros((2, 4, 84, 84, 1), dtype=jnp.uint8)))
        assert q.shape == (2, 3)


class TestDuelingDQNGradient:
    def test_dueling_network_output_shape(self):
        net = DuelingQNetwork(action_dim=4)
        params = net.init(jax.random.key(0), jnp.zeros((8,)))
        q = cast(jax.Array, net.apply(params, jnp.ones((8,))))
        assert q.shape == (4,)

    def test_dueling_batch_shape(self):
        net = DuelingQNetwork(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        q = cast(jax.Array, net.apply(params, jnp.ones((10, 4))))
        assert q.shape == (10, 3)

    def test_dueling_params_change_after_update(self):
        net = DuelingQNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))

        tx = optax.adam(3e-4)
        state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        obs = jax.random.normal(jax.random.key(1), (16, 4))
        targets = jax.random.normal(jax.random.key(2), (16, 2))

        def loss_fn(params):
            q = cast(jax.Array, net.apply(params, obs))
            return jnp.mean(jnp.square(q - targets))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))

    def test_dueling_jit(self):
        net = DuelingQNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))

        @jax.jit
        def forward(params, x):
            return net.apply(params, x)

        q = forward(params, jnp.ones((4,)))
        assert q.shape == (2,)

    def test_dueling_network_uses_mlp_by_default(self):
        net = _make_dueling_q_network(DuelingDQNConfig(), action_dim=2)
        assert isinstance(net, DuelingQNetwork)

    def test_dueling_network_rejects_nature_preset_until_specified(self):
        with pytest.raises(ValueError, match="not yet supported"):
            _make_dueling_q_network(DuelingDQNConfig(NETWORK_PRESET="nature_cnn"), action_dim=2)
