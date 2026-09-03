"""Small tests for rl_components.buffers — shapes, pointer wrap, sampling."""

import jax
import jax.numpy as jnp
from rl_components.buffers import ReplayBuffer, ReplayBufferState


class TestReplayBufferInit:
    def test_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=())
        state = buf.init()
        assert state.obs.shape == (100, 4)
        assert state.actions.shape == (100,)
        assert state.rewards.shape == (100,)
        assert state.next_obs.shape == (100, 4)
        assert state.discount.shape == (100,)

    def test_pointer_starts_at_zero(self) -> None:
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=(1,))
        state = buf.init()
        assert int(state.pointer) == 0
        assert int(state.count) == 0

    def test_action_dtype(self) -> None:
        buf = ReplayBuffer(capacity=5, obs_shape=(3,), action_shape=(), action_dtype=jnp.int32)
        state = buf.init()
        assert state.actions.dtype == jnp.int32

    def test_obs_dtype_defaults_to_float32_but_can_be_configured(self) -> None:
        default_state = ReplayBuffer(capacity=5, obs_shape=(3,), action_shape=()).init()
        assert default_state.obs.dtype == jnp.float32
        assert default_state.next_obs.dtype == jnp.float32

        configured_state = ReplayBuffer(capacity=5, obs_shape=(3,), action_shape=(), obs_dtype=jnp.uint8).init()
        assert configured_state.obs.dtype == jnp.uint8
        assert configured_state.next_obs.dtype == jnp.uint8

    def test_discount_dtype_is_float32(self) -> None:
        buf = ReplayBuffer(capacity=5, obs_shape=(2,), action_shape=())
        state = buf.init()
        assert state.discount.dtype == jnp.float32


class TestReplayBufferAdd:
    def test_single_add(self) -> None:
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=())
        state = buf.init()
        state = buf.add(
            state,
            obs=jnp.array([[1.0, 2.0]]),
            action=jnp.array([0.0]),
            reward=jnp.array([1.0]),
            next_obs=jnp.array([[3.0, 4.0]]),
            discount=jnp.array([0.99]),
        )
        assert int(state.pointer) == 1
        assert int(state.count) == 1
        assert jnp.allclose(state.obs[0], jnp.array([1.0, 2.0]))

    def test_batch_add(self) -> None:
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=())
        state = buf.init()
        state = buf.add(
            state,
            obs=jnp.ones((3, 2)),
            action=jnp.zeros((3,)),
            reward=jnp.ones((3,)),
            next_obs=jnp.ones((3, 2)),
            discount=jnp.full((3,), 0.99),
        )
        assert int(state.pointer) == 3
        assert int(state.count) == 3

    def test_pointer_wraps(self) -> None:
        buf = ReplayBuffer(capacity=4, obs_shape=(1,), action_shape=())
        state = buf.init()
        for _ in range(5):
            state = buf.add(
                state,
                obs=jnp.ones((1, 1)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 1)),
                discount=jnp.full((1,), 0.99),
            )
        assert int(state.pointer) == 1  # 5 % 4 = 1
        assert int(state.count) == 4  # capped at capacity

    def test_count_caps_at_capacity(self) -> None:
        buf = ReplayBuffer(capacity=3, obs_shape=(1,), action_shape=())
        state = buf.init()
        for _ in range(10):
            state = buf.add(
                state,
                obs=jnp.ones((1, 1)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 1)),
                discount=jnp.full((1,), 0.99),
            )
        assert int(state.count) == 3


class TestReplayBufferSample:
    def test_sample_shape(self) -> None:
        buf = ReplayBuffer(capacity=10, obs_shape=(4,), action_shape=())
        state = buf.init()
        for _ in range(10):
            state = buf.add(
                state,
                obs=jnp.ones((1, 4)),
                action=jnp.zeros((1,)),
                reward=jnp.ones((1,)),
                next_obs=jnp.ones((1, 4)),
                discount=jnp.full((1,), 0.99),
            )
        key = jax.random.key(0)
        obs, actions, rewards, next_obs, discounts = buf.sample(state, key, batch_size=5)
        assert obs.shape == (5, 4)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_obs.shape == (5, 4)
        assert discounts.shape == (5,)

    def test_sample_returns_stored_data(self) -> None:
        buf = ReplayBuffer(capacity=1, obs_shape=(2,), action_shape=())
        state = buf.init()
        state = buf.add(
            state,
            obs=jnp.array([[42.0, 43.0]]),
            action=jnp.array([7.0]),
            reward=jnp.array([99.0]),
            next_obs=jnp.array([[44.0, 45.0]]),
            discount=jnp.array([0.0]),
        )
        key = jax.random.key(0)
        obs, actions, rewards, next_obs, discounts = buf.sample(state, key, batch_size=1)
        assert jnp.allclose(obs[0], jnp.array([42.0, 43.0]))
        assert float(rewards[0]) == 99.0


class TestReplayBufferStateNamedTuple:
    def test_is_named_tuple(self) -> None:
        buf = ReplayBuffer(capacity=5, obs_shape=(1,), action_shape=())
        state = buf.init()
        assert isinstance(state, ReplayBufferState)
        assert isinstance(state, tuple)
