"""Medium tests for rl_components.buffers — JIT compilation and functional correctness."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from rl_components.buffers import ReplayBuffer


class TestReplayBufferJIT:
    def test_add_jittable(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(4,), action_shape=())
        state = buf.init()

        @jax.jit
        def add_one(state):
            return buf.add(
                state,
                obs=jnp.ones((1, 4)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 4)),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )

        state = add_one(state)
        assert int(state.pointer) == 1
        assert int(state.count) == 1

    def test_sample_jittable(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(4,), action_shape=())
        state = buf.init()
        for _ in range(10):
            state = buf.add(
                state,
                obs=jnp.ones((1, 4)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 4)),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )

        @jax.jit
        def sample(state, key):
            return buf.sample(state, key, batch_size=4)

        key = jax.random.key(0)
        obs, actions, rewards, next_obs, dones = sample(state, key)
        assert obs.shape == (4, 4)

    def test_add_sample_loop_in_scan(self):
        """Buffer add + sample works inside lax.scan."""
        buf = ReplayBuffer(capacity=50, obs_shape=(2,), action_shape=(), action_dtype=jnp.int32)
        state = buf.init()

        def step(carry, t):
            buf_state, key = carry
            key, k1 = jax.random.split(key)
            buf_state = buf.add(
                buf_state,
                obs=jnp.ones((1, 2)) * t,
                action=jnp.zeros((1,), dtype=jnp.int32),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 2)),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )
            obs, _, _, _, _ = buf.sample(buf_state, k1, batch_size=2)
            return (buf_state, key), obs.mean()

        key = jax.random.key(42)
        (final_state, _), means = jax.jit(lambda s, k: jax.lax.scan(step, (s, k), jnp.arange(20)))(state, key)
        assert int(final_state.count) == 20
        assert means.shape == (20,)

    def test_add_and_sample_uint8_observations_under_jit(self):
        buf = ReplayBuffer(capacity=8, obs_shape=(2, 2, 1), action_shape=(), action_dtype=jnp.int32, obs_dtype=jnp.uint8)
        state = buf.init()

        @jax.jit
        def add_and_sample(state, key):
            updated_state = buf.add(
                state,
                obs=jnp.full((1, 2, 2, 1), 7, dtype=jnp.uint8),
                action=jnp.zeros((1,), dtype=jnp.int32),
                reward=jnp.ones((1,)),
                next_obs=jnp.full((1, 2, 2, 1), 9, dtype=jnp.uint8),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )
            return updated_state, buf.sample(updated_state, key, batch_size=1)

        next_state, sample = add_and_sample(state, jax.random.key(0))
        obs, actions, rewards, next_obs, dones = sample

        assert next_state.obs.dtype == jnp.uint8
        assert next_state.next_obs.dtype == jnp.uint8
        assert obs.dtype == jnp.uint8
        assert next_obs.dtype == jnp.uint8
        assert actions.dtype == jnp.int32
        assert rewards.dtype == jnp.float32
        assert dones.dtype == jnp.bool_


class TestReplayBufferNetworkShapes:
    def test_network_forward_with_sampled_obs(self):
        """Sampled observations have correct shape for a simple network forward pass."""
        import flax.linen as nn

        class SimpleNet(nn.Module):
            if TYPE_CHECKING:
                def apply(
                    self,
                    variables: object,
                    x: jax.Array,
                    *,
                    rngs: object | None = None,
                ) -> jax.Array: ...

            @nn.compact
            def __call__(self, x):
                return nn.Dense(2)(x)

        buf = ReplayBuffer(capacity=10, obs_shape=(4,), action_shape=())
        state = buf.init()
        for _ in range(5):
            state = buf.add(
                state,
                obs=jnp.ones((1, 4)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 4)),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )

        net = SimpleNet()
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        key = jax.random.key(1)
        obs, _, _, _, _ = buf.sample(state, key, batch_size=3)
        out = net.apply(params, obs)
        assert out.shape == (3, 2)
