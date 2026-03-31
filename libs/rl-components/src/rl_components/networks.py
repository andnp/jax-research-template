from typing import TYPE_CHECKING, cast

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from rl_components.structs import chex_struct


def _activation_fn(name: str):
    if name == "relu":
        return nn.relu
    return nn.tanh


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

    def log_prob(self, value: jax.Array) -> jax.Array:
        clipped_value = jnp.clip(value, -1.0 + self.epsilon, 1.0 - self.epsilon)
        pre_tanh_value = jnp.arctanh(clipped_value)
        log_prob = self._base_distribution().log_prob(pre_tanh_value)
        return cast(jax.Array, log_prob - jnp.log(1.0 - jnp.square(clipped_value) + self.epsilon))

    def entropy(self) -> jax.Array:
        return cast(jax.Array, self._base_distribution().entropy())


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    if TYPE_CHECKING:
        def apply(
            self,
            variables: object,
            x: jax.Array,
            *,
            rngs: object | None = None,
        ) -> tuple[distrax.Categorical, jax.Array]: ...

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, jnp.ndarray]:
        activation = _activation_fn(self.activation)

        # Separate actor and critic paths for stability
        actor_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor_x = activation(actor_x)
        actor_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor_x)
        actor_x = activation(actor_x)
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor_x)
        probs = distrax.Categorical(logits=actor_mean)

        critic_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic_x = activation(critic_x)
        critic_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic_x)
        critic_x = activation(critic_x)
        critic_value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic_x)

        return probs, jnp.squeeze(critic_value, axis=-1)


class ContinuousActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    if TYPE_CHECKING:
        def apply(
            self,
            variables: object,
            x: jax.Array,
            *,
            rngs: object | None = None,
        ) -> tuple[TanhNormalDiag, jax.Array]: ...

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[TanhNormalDiag, jnp.ndarray]:
        activation = _activation_fn(self.activation)

        actor_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor_x = activation(actor_x)
        actor_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor_x)
        actor_x = activation(actor_x)
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor_x)
        actor_log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        actor_log_std = jnp.clip(actor_log_std, self.log_std_min, self.log_std_max)
        policy = TanhNormalDiag(mean=actor_mean, log_std=actor_log_std)

        critic_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic_x = activation(critic_x)
        critic_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic_x)
        critic_x = activation(critic_x)
        critic_value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic_x)

        return policy, jnp.squeeze(critic_value, axis=-1)
