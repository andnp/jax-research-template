"""Fixed-architecture actor-critic networks tied to the PPO/on-policy paper convention."""

from __future__ import annotations

from typing import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_nn.typed_module import TypedApply
from rl_components.networks import TanhNormalDiag


def _activation_fn(name: str) -> Callable[[jax.Array], jax.Array]:
    if name == "relu":
        return nn.relu
    return nn.tanh


class ActorCritic(TypedApply[tuple[distrax.Categorical, jax.Array]], nn.Module):
    action_dim: int
    activation: str = "tanh"

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


class ContinuousActorCritic(TypedApply[tuple[TanhNormalDiag, jax.Array]], nn.Module):
    action_dim: int
    activation: str = "tanh"
    log_std_min: float = -20.0
    log_std_max: float = 2.0

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


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)

    def apply(
        self,
        variables: object,
        x: jax.Array,
        a: jax.Array,
        *,
        rngs: object | None = None,
    ) -> jax.Array:
        return super().apply(variables, x, a, rngs=rngs)
