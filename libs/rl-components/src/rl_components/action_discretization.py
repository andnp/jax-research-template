"""Adapt two-action discrete environments for continuous-action agents."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep


class DiscretizeAction[StateT, ParamsT]:
    """Present a two-action discrete environment through scalar continuous actions."""

    def __init__(self, env: EnvProtocol[jax.Array, StateT, jax.Array, ParamsT]) -> None:
        self._env = env

    def spec(self, params: ParamsT | None = None) -> EnvSpec:
        inner = self._env.spec(params)
        if inner.num_actions != 2:
            raise ValueError(
                f"DiscretizeAction requires a two-action environment, got spec {inner.id!r} with {inner.num_actions}"
            )
        return EnvSpec(
            id=inner.id,
            observation_shape=tuple(inner.observation_shape),
            action_shape=(1,),
            observation_dtype=jnp.dtype(inner.observation_dtype),
            action_dtype=jnp.dtype(jnp.float32),
            action_low=jnp.full((1,), -1.0, dtype=jnp.float32),
            action_high=jnp.full((1,), 1.0, dtype=jnp.float32),
            truncation_policy=inner.truncation_policy,
        )

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None) -> EnvReset[jax.Array, StateT]:
        return self._env.reset(key, params)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: jax.Array,
        params: ParamsT | None = None,
    ) -> EnvStep[jax.Array, StateT]:
        discrete_action = jnp.where(action[0] < 0.0, 0, 1).astype(jnp.int32)
        return self._env.step(key, state, discrete_action, params)
