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
