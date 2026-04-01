"""Medium tests for rl_components.networks — ActorCritic forward pass and JIT."""

import jax
import jax.numpy as jnp
from rl_components.networks import ActorCritic, ContinuousActorCritic


class TestActorCriticForward:
    def test_output_shapes(self):
        net = ActorCritic(action_dim=4)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((8,)))
        probs, value = net.apply(params, jnp.ones((8,)))
        assert probs.logits.shape == (4,)
        assert value.shape == ()

    def test_batch_output_shapes(self):
        net = ActorCritic(action_dim=3)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))
        probs, value = net.apply(params, jnp.ones((5, 4)))
        assert probs.logits.shape == (5, 3)
        assert value.shape == (5,)

    def test_activation_relu(self):
        net = ActorCritic(action_dim=2, activation="relu")
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))
        probs, value = net.apply(params, jnp.ones((4,)))
        assert probs.logits.shape == (2,)

    def test_jit_compilation(self):
        net = ActorCritic(action_dim=2)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))

        @jax.jit
        def forward(params, x):
            return net.apply(params, x)

        probs, value = forward(params, jnp.ones((4,)))
        assert probs.logits.shape == (2,)
        assert value.shape == ()

    def test_probs_sum_to_one(self):
        net = ActorCritic(action_dim=3)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))
        probs, _ = net.apply(params, jnp.ones((4,)))
        prob_sum = jnp.exp(probs.logits).sum()
        assert jnp.allclose(prob_sum, 1.0, atol=1e-5)

    def test_value_is_scalar_per_sample(self):
        net = ActorCritic(action_dim=2)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))
        _, value = net.apply(params, jnp.ones((10, 4)))
        assert value.shape == (10,)


class TestContinuousActorCriticForward:
    def test_output_shapes_and_action_bounds(self):
        net = ContinuousActorCritic(action_dim=3)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((8,)))
        policy, value = net.apply(params, jnp.ones((8,)))
        action = policy.sample(seed=jax.random.key(1))
        log_prob = jnp.sum(policy.log_prob(action), axis=-1)
        entropy = jnp.sum(policy.entropy(), axis=-1)

        assert policy.mean.shape == (3,)
        assert action.shape == (3,)
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <= 1.0)
        assert log_prob.shape == ()
        assert entropy.shape == ()
        assert value.shape == ()

    def test_batch_output_shapes_reduce_per_sample(self):
        net = ContinuousActorCritic(action_dim=2)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))
        policy, value = net.apply(params, jnp.ones((5, 4)))
        action = policy.sample(seed=jax.random.key(1))
        log_prob = jnp.sum(policy.log_prob(action), axis=-1)
        entropy = jnp.sum(policy.entropy(), axis=-1)

        assert policy.mean.shape == (5, 2)
        assert action.shape == (5, 2)
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5,)
