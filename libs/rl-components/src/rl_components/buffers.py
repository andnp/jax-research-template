from typing import NamedTuple

import jax
import jax.numpy as jnp


class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    discount: jnp.ndarray
    pointer: jnp.ndarray
    count: jnp.ndarray


class FrameReplayBufferState(NamedTuple):
    frame: jnp.ndarray
    next_frame: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    discount: jnp.ndarray
    episode_end: jnp.ndarray
    pointer: jnp.ndarray
    count: jnp.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: jnp.dtype = jnp.float32,
        obs_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.obs_dtype = obs_dtype

    @classmethod
    def from_state(cls, state: ReplayBufferState) -> "ReplayBuffer":
        """Rebuild a buffer from the geometry recorded in one of its own states.

        The buffer object holds no data, only shapes and dtypes, and every one of them is
        recoverable from ``state``. An agent whose ``step`` is closed over by ``jit``
        therefore needs nothing extra in its carry to reach its buffer: the state it
        already carries is the geometry.
        """
        return cls(
            capacity=state.obs.shape[0],
            obs_shape=state.obs.shape[1:],
            action_shape=state.actions.shape[1:],
            action_dtype=state.actions.dtype,
            obs_dtype=state.obs.dtype,
        )

    def init(self) -> ReplayBufferState:
        return ReplayBufferState(
            obs=jnp.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype),
            actions=jnp.zeros((self.capacity,) + self.action_shape, dtype=self.action_dtype),
            rewards=jnp.zeros((self.capacity,)),
            next_obs=jnp.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype),
            discount=jnp.zeros((self.capacity,), dtype=jnp.float32),
            pointer=jnp.array(0),
            count=jnp.array(0),
        )

    def add(self, state: ReplayBufferState, obs: jax.Array, action: jax.Array, reward: jax.Array, next_obs: jax.Array, discount: jax.Array) -> ReplayBufferState:
        # Vectorized add for multiple envs
        num_to_add = obs.shape[0]
        indices = (state.pointer + jnp.arange(num_to_add)) % self.capacity

        obs_new = state.obs.at[indices].set(obs)
        actions_new = state.actions.at[indices].set(action)
        rewards_new = state.rewards.at[indices].set(reward)
        next_obs_new = state.next_obs.at[indices].set(next_obs)
        discount_new = state.discount.at[indices].set(discount)

        return state._replace(
            obs=obs_new,
            actions=actions_new,
            rewards=rewards_new,
            next_obs=next_obs_new,
            discount=discount_new,
            pointer=(state.pointer + num_to_add) % self.capacity,
            count=jnp.minimum(state.count + num_to_add, self.capacity),
        )

    def sample(self, state: ReplayBufferState, key: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        indices = jax.random.randint(key, (batch_size,), 0, state.count)
        return (
            state.obs[indices],
            state.actions[indices],
            state.rewards[indices],
            state.next_obs[indices],
            state.discount[indices],
        )


class FrameReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: jnp.dtype = jnp.float32,
        obs_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if len(obs_shape) < 1:
            raise ValueError("FrameReplayBuffer observations must have a frame axis")
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.obs_dtype = obs_dtype
        self.n_frames = obs_shape[0]
        self.frame_shape = obs_shape[1:]

    @classmethod
    def from_state(cls, state: FrameReplayBufferState, n_frames: int) -> "FrameReplayBuffer":
        return cls(
            capacity=state.frame.shape[0],
            obs_shape=(n_frames,) + state.frame.shape[1:],
            action_shape=state.actions.shape[1:],
            action_dtype=state.actions.dtype,
            obs_dtype=state.frame.dtype,
        )

    def init(self) -> FrameReplayBufferState:
        return FrameReplayBufferState(
            frame=jnp.zeros((self.capacity,) + self.frame_shape, dtype=self.obs_dtype),
            next_frame=jnp.zeros((self.capacity,) + self.frame_shape, dtype=self.obs_dtype),
            actions=jnp.zeros((self.capacity,) + self.action_shape, dtype=self.action_dtype),
            rewards=jnp.zeros((self.capacity,)),
            discount=jnp.zeros((self.capacity,), dtype=jnp.float32),
            episode_end=jnp.zeros((self.capacity,), dtype=jnp.bool_),
            pointer=jnp.array(0),
            count=jnp.array(0),
        )

    def add(
        self,
        state: FrameReplayBufferState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        discount: jax.Array,
        episode_end: jax.Array,
    ) -> FrameReplayBufferState:
        if obs.shape[0] != 1:
            raise ValueError("FrameReplayBuffer.add accepts exactly one transition")
        index = state.pointer
        return state._replace(
            frame=state.frame.at[index].set(obs[0, -1]),
            next_frame=state.next_frame.at[index].set(next_obs[0, -1]),
            actions=state.actions.at[index].set(action[0]),
            rewards=state.rewards.at[index].set(reward[0]),
            discount=state.discount.at[index].set(discount[0]),
            episode_end=state.episode_end.at[index].set(episode_end[0]),
            pointer=(state.pointer + 1) % self.capacity,
            count=jnp.minimum(state.count + 1, self.capacity),
        )

    def _episode_start(self, state: FrameReplayBufferState, index: jax.Array) -> jax.Array:
        oldest = (state.pointer - state.count) % self.capacity
        distance = (index - oldest) % self.capacity
        start = (index - jnp.minimum(self.n_frames - 1, distance)) % self.capacity
        for offset in range(self.n_frames - 1, -1, -1):
            predecessor = (index - offset - 1) % self.capacity
            candidate = (index - offset) % self.capacity
            is_available = distance >= offset
            is_first = offset == distance
            is_boundary = is_available & (is_first | state.episode_end[predecessor])
            start = jnp.where(is_boundary, candidate, start)
        return start

    def _observation(self, state: FrameReplayBufferState, index: jax.Array) -> jax.Array:
        start = self._episode_start(state, index)
        offsets = jnp.arange(self.n_frames - 1, -1, -1)
        distances = jnp.minimum(offsets, (index - start) % self.capacity)
        indices = (index - distances) % self.capacity
        return state.frame[indices]

    def sample(self, state: FrameReplayBufferState, key: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        minimum_age = jnp.where(state.count == self.capacity, self.n_frames - 1, 0)
        offsets = jax.random.randint(key, (batch_size,), minimum_age, state.count)
        oldest = (state.pointer - state.count) % self.capacity
        indices = (oldest + offsets) % self.capacity
        def observation(index: jax.Array) -> jax.Array:
            return self._observation(state, index)

        def next_observation(index: jax.Array) -> jax.Array:
            return jnp.concatenate((self._observation(state, index)[1:], state.next_frame[index][None]), axis=0)

        observations = jax.vmap(observation)(indices)
        next_observations = jax.vmap(next_observation)(indices)
        return (
            observations,
            state.actions[indices],
            state.rewards[indices],
            next_observations,
            state.discount[indices],
        )
