from typing import cast

import distrax
import jax
import jax.numpy as jnp

from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class TanhNormalDiag:
    mean: jax.Array
    log_std: jax.Array
    epsilon: float = 1e-6

    def _base_distribution(self) -> distrax.Normal:
        return distrax.Normal(self.mean, jnp.exp(self.log_std))

    def sample(self, seed: jax.Array) -> jax.Array:
        raw_action = self._base_distribution().sample(seed=seed)
        return cast(jax.Array, jnp.tanh(raw_action))

    def sample_and_log_prob(self, seed: jax.Array) -> tuple[jax.Array, jax.Array]:
        raw_action = self._base_distribution().sample(seed=seed)
        action = jnp.tanh(raw_action)
        log_prob = self._base_distribution().log_prob(raw_action)
        log_prob -= jnp.log(1.0 - jnp.square(action) + self.epsilon)
        return action, log_prob

    def log_prob(self, value: jax.Array) -> jax.Array:
        clipped_value = jnp.clip(value, -1.0 + self.epsilon, 1.0 - self.epsilon)
        pre_tanh_value = jnp.arctanh(clipped_value)
        log_prob = self._base_distribution().log_prob(pre_tanh_value)
        return cast(jax.Array, log_prob - jnp.log(1.0 - jnp.square(clipped_value) + self.epsilon))

    def entropy(self) -> jax.Array:
        return cast(jax.Array, self._base_distribution().entropy())
