from typing import NamedTuple

import jax
import jax.numpy as jnp


class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    pointer: jnp.ndarray
    count: jnp.ndarray  # pyright: ignore[reportIncompatibleMethodOverride]


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: jnp.dtype = jnp.float32,
        obs_dtype: jnp.dtype = jnp.float32,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.obs_dtype = obs_dtype

    def init(self) -> ReplayBufferState:
        return ReplayBufferState(
            obs=jnp.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype),
            actions=jnp.zeros((self.capacity,) + self.action_shape, dtype=self.action_dtype),
            rewards=jnp.zeros((self.capacity,)),
            next_obs=jnp.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype),
            dones=jnp.zeros((self.capacity,), dtype=jnp.bool_),
            pointer=jnp.array(0),
            count=jnp.array(0),
        )

    def add(self, state: ReplayBufferState, obs, action, reward, next_obs, done) -> ReplayBufferState:
        # Vectorized add for multiple envs
        num_to_add = obs.shape[0]
        indices = (state.pointer + jnp.arange(num_to_add)) % self.capacity

        obs_new = state.obs.at[indices].set(obs)
        actions_new = state.actions.at[indices].set(action)
        rewards_new = state.rewards.at[indices].set(reward)
        next_obs_new = state.next_obs.at[indices].set(next_obs)
        dones_new = state.dones.at[indices].set(done)

        return state._replace(
            obs=obs_new,
            actions=actions_new,
            rewards=rewards_new,
            next_obs=next_obs_new,
            dones=dones_new,
            pointer=(state.pointer + num_to_add) % self.capacity,
            count=jnp.minimum(state.count + num_to_add, self.capacity),
        )

    def sample(self, state: ReplayBufferState, key, batch_size: int):
        indices = jax.random.randint(key, (batch_size,), 0, state.count)
        return (
            state.obs[indices],
            state.actions[indices],
            state.rewards[indices],
            state.next_obs[indices],
            state.dones[indices],
        )
