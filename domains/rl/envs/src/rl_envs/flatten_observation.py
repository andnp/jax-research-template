"""Observation-shape adapter for vector policies."""

from __future__ import annotations

from math import prod

import chex
import jax
import jax.numpy as jnp
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep


class FlattenObservationWrapper[ObsT: jax.Array, StateT, ActionT, ParamsT]:
    def __init__(self, env: EnvProtocol[ObsT, StateT, ActionT, ParamsT]) -> None:
        self._env = env

    def spec(self, params: ParamsT | None = None):
        inner_spec = self._env.spec(params)
        return EnvSpec(
            id=inner_spec.id,
            observation_shape=(prod(inner_spec.observation_shape),),
            action_shape=inner_spec.action_shape,
            observation_dtype=inner_spec.observation_dtype,
            action_dtype=inner_spec.action_dtype,
            num_actions=inner_spec.num_actions,
            action_low=inner_spec.action_low,
            action_high=inner_spec.action_high,
            truncation_policy=inner_spec.truncation_policy,
        )

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None):
        reset = self._env.reset(key, params)
        return EnvReset(observation=jnp.reshape(reset.observation, self.spec(params).observation_shape), state=reset.state)

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: ActionT,
        params: ParamsT | None = None,
    ):
        step = self._env.step(key, state, action, params)
        return EnvStep(
            observation=jnp.reshape(step.observation, self.spec(params).observation_shape),
            state=step.state,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=step.info,
        )
