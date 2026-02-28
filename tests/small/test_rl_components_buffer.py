"""Small tests for rl_components.buffers — shapes, pointer wrap, sampling."""

import jax
import jax.numpy as jnp
from rl_components.buffers import ReplayBuffer, ReplayBufferState


class TestReplayBufferInit:
    def test_shapes(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=())
        state = buf.init()
        assert state.obs.shape == (100, 4)
        assert state.actions.shape == (100,)
        assert state.rewards.shape == (100,)
        assert state.next_obs.shape == (100, 4)
        assert state.dones.shape == (100,)

    def test_pointer_starts_at_zero(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=(1,))
        state = buf.init()
        assert int(state.pointer) == 0
        assert int(state.count) == 0

    def test_action_dtype(self):
        buf = ReplayBuffer(capacity=5, obs_shape=(3,), action_shape=(), action_dtype=jnp.int32)
        state = buf.init()
        assert state.actions.dtype == jnp.int32

    def test_dones_dtype_is_bool(self):
        buf = ReplayBuffer(capacity=5, obs_shape=(2,), action_shape=())
        state = buf.init()
        assert state.dones.dtype == jnp.bool_


class TestReplayBufferAdd:
    def test_single_add(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=())
        state = buf.init()
        state = buf.add(
            state,
            obs=jnp.array([[1.0, 2.0]]),
            action=jnp.array([0.0]),
            reward=jnp.array([1.0]),
            next_obs=jnp.array([[3.0, 4.0]]),
            done=jnp.array([False]),
        )
        assert int(state.pointer) == 1
        assert int(state.count) == 1
        assert jnp.allclose(state.obs[0], jnp.array([1.0, 2.0]))

    def test_batch_add(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=())
        state = buf.init()
        state = buf.add(
            state,
            obs=jnp.ones((3, 2)),
            action=jnp.zeros((3,)),
            reward=jnp.ones((3,)),
            next_obs=jnp.ones((3, 2)),
            done=jnp.zeros((3,), dtype=jnp.bool_),
        )
        assert int(state.pointer) == 3
        assert int(state.count) == 3

    def test_pointer_wraps(self):
        buf = ReplayBuffer(capacity=4, obs_shape=(1,), action_shape=())
        state = buf.init()
        for _ in range(5):
            state = buf.add(
                state,
                obs=jnp.ones((1, 1)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 1)),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )
        assert int(state.pointer) == 1  # 5 % 4 = 1
        assert int(state.count) == 4  # capped at capacity

    def test_count_caps_at_capacity(self):
        buf = ReplayBuffer(capacity=3, obs_shape=(1,), action_shape=())
        state = buf.init()
        for _ in range(10):
            state = buf.add(
                state,
                obs=jnp.ones((1, 1)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 1)),
                done=jnp.zeros((1,), dtype=jnp.bool_),
            )
        assert int(state.count) == 3


class TestReplayBufferSample:
    def test_sample_shape(self):
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
        key = jax.random.key(0)
        obs, actions, rewards, next_obs, dones = buf.sample(state, key, batch_size=5)
        assert obs.shape == (5, 4)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_obs.shape == (5, 4)
        assert dones.shape == (5,)

    def test_sample_returns_stored_data(self):
        buf = ReplayBuffer(capacity=1, obs_shape=(2,), action_shape=())
        state = buf.init()
        state = buf.add(
            state,
            obs=jnp.array([[42.0, 43.0]]),
            action=jnp.array([7.0]),
            reward=jnp.array([99.0]),
            next_obs=jnp.array([[44.0, 45.0]]),
            done=jnp.array([True]),
        )
        key = jax.random.key(0)
        obs, actions, rewards, next_obs, dones = buf.sample(state, key, batch_size=1)
        assert jnp.allclose(obs[0], jnp.array([42.0, 43.0]))
        assert float(rewards[0]) == 99.0


class TestReplayBufferStateNamedTuple:
    def test_is_named_tuple(self):
        buf = ReplayBuffer(capacity=5, obs_shape=(1,), action_shape=())
        state = buf.init()
        assert isinstance(state, ReplayBufferState)
        assert isinstance(state, tuple)
