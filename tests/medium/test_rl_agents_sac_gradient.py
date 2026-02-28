"""Medium tests for rl_agents.sac — gradient flow and JIT compilation."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_agents.sac import Actor, Critic


class TestSACGradientFlow:
    def test_critic_params_change_after_update(self):
        """Critic parameters should change after a gradient step."""
        critic = Critic()
        obs_dim = (4,)
        action_dim = 2
        params = critic.init(jax.random.key(0), jnp.zeros(obs_dim), jnp.zeros((action_dim,)))

        tx = optax.adam(3e-4)
        state = TrainState.create(apply_fn=critic.apply, params=params, tx=tx)

        obs = jax.random.normal(jax.random.key(1), (16, 4))
        actions = jax.random.normal(jax.random.key(2), (16, 2))
        targets = jax.random.normal(jax.random.key(3), (16,))

        def loss_fn(params):
            q = critic.apply(params, obs, actions)
            return jnp.mean(jnp.square(q - targets))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed

    def test_actor_params_change_after_update(self):
        """Actor parameters should change after a gradient step."""
        actor = Actor(action_dim=2)
        obs_dim = (4,)
        params = actor.init(jax.random.key(0), jnp.zeros(obs_dim))

        tx = optax.adam(3e-4)
        state = TrainState.create(apply_fn=actor.apply, params=params, tx=tx)

        obs = jax.random.normal(jax.random.key(1), (16, 4))

        def loss_fn(params):
            actions, log_probs = jax.vmap(
                lambda o, k: actor.sample(params, o, k),
                in_axes=(0, 0),
            )(obs, jax.random.split(jax.random.key(2), 16))
            return jnp.mean(log_probs)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed

    def test_critic_jit(self):
        """Critic forward pass should JIT-compile."""
        critic = Critic()
        params = critic.init(jax.random.key(0), jnp.zeros((4,)), jnp.zeros((2,)))

        @jax.jit
        def forward(params, obs, action):
            return critic.apply(params, obs, action)

        q = forward(params, jnp.ones((4,)), jnp.ones((2,)))
        assert q.shape == ()

    def test_actor_sample_jit(self):
        """Actor sampling should JIT-compile."""
        actor = Actor(action_dim=2)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))

        @jax.jit
        def sample(params, obs, key):
            return actor.sample(params, obs, key)

        action, log_prob = sample(params, jnp.ones((4,)), jax.random.key(1))
        assert action.shape == (2,)
        assert log_prob.shape == ()
