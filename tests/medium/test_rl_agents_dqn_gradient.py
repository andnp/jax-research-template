"""Medium tests for rl_agents.dqn — gradient flow and JIT compilation."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_agents.dqn import QNetwork


class TestDQNGradientFlow:
    def test_params_change_after_update(self):
        """Network parameters should change after a gradient step."""
        net = QNetwork(action_dim=2)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))

        tx = optax.adam(3e-4)
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        # Fake batch
        obs = jax.random.normal(jax.random.key(1), (32, 4))
        actions = jax.random.randint(jax.random.key(2), (32,), 0, 2)
        rewards = jax.random.normal(jax.random.key(3), (32,))
        next_obs = jax.random.normal(jax.random.key(4), (32, 4))
        dones = jnp.zeros((32,))
        target_params = params

        def loss_fn(params):
            q_values = net.apply(params, obs)
            q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()
            next_q = net.apply(target_params, next_obs)
            next_q_max = jnp.max(next_q, axis=-1)
            target = rewards + 0.99 * next_q_max * (1.0 - dones)
            return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(train_state.params)
        new_state = train_state.apply_gradients(grads=grads)

        # Params should differ
        old_flat = jax.tree_util.tree_leaves(train_state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed, "Parameters did not change after gradient step"
        assert float(loss) > 0

    def test_loss_fn_jit(self):
        """Loss function should be JIT-compilable."""
        net = QNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))

        @jax.jit
        def compute_loss(params, obs, actions, rewards, next_obs, dones):
            q_values = net.apply(params, obs)
            q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()
            next_q = net.apply(params, next_obs)
            next_q_max = jnp.max(next_q, axis=-1)
            target = rewards + 0.99 * next_q_max * (1.0 - dones)
            return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))

        obs = jnp.ones((8, 4))
        actions = jnp.zeros((8,), dtype=jnp.int32)
        rewards = jnp.ones((8,))
        next_obs = jnp.ones((8, 4))
        dones = jnp.zeros((8,))

        loss = compute_loss(params, obs, actions, rewards, next_obs, dones)
        assert loss.shape == ()
