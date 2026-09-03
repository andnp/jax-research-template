"""Minimal Gymnax compatibility bridge for canonical environments."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvProtocol, EnvSpec
from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class GymnaxSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


@chex_struct(frozen=True)
class GymnaxDiscreteSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    n: int
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


def _observation_space_from_spec(spec: EnvSpec) -> GymnaxSpace:
    return GymnaxSpace(
        shape=tuple(spec.observation_shape),
        dtype=jnp.dtype(spec.observation_dtype),
    )


def _action_space_from_spec(spec: EnvSpec) -> GymnaxSpace | GymnaxDiscreteSpace:
    if spec.num_actions is not None:
        return GymnaxDiscreteSpace(
            shape=tuple(spec.action_shape),
            dtype=jnp.dtype(spec.action_dtype),
            n=int(spec.num_actions),
        )
    return GymnaxSpace(
        shape=tuple(spec.action_shape),
        dtype=jnp.dtype(spec.action_dtype),
        action_low=spec.action_low,
        action_high=spec.action_high,
    )


class GymnaxCompatibilityBridge[ObservationT, StateT, ActionT, ParamsT]:
    def __init__(self, env: EnvProtocol[ObservationT, StateT, ActionT, ParamsT]) -> None:
        self._env = env

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)

    def observation_space(self, params: object | None = None) -> GymnaxSpace:
        return _observation_space_from_spec(self._env.spec(cast(ParamsT | None, params)))

    def action_space(self, params: object | None = None) -> GymnaxSpace | GymnaxDiscreteSpace:
        return _action_space_from_spec(self._env.spec(cast(ParamsT | None, params)))

    def reset(self, key: chex.PRNGKey, params: object | None = None) -> tuple[ObservationT, StateT]:
        reset = self._env.reset(key, cast(ParamsT | None, params))
        return reset.observation, reset.state

    def step(
        self,
        key: chex.PRNGKey,
        state: object,
        action: ActionT,
        params: object | None = None,
    ) -> tuple[ObservationT, object, jax.Array, jax.Array, dict[str, jax.Array]]:
        transition = self._env.step(key, cast(StateT, state), action, cast(ParamsT | None, params))
        done = jnp.logical_or(transition.terminated, transition.truncated)
        info = dict(transition.info)
        info.setdefault("terminated", jnp.asarray(transition.terminated))
        info.setdefault("truncated", jnp.asarray(transition.truncated))
        return transition.observation, transition.state, transition.reward, done, info


def make_gymnax_compat_env[ObservationT, StateT, ActionT, ParamsT](
    env: EnvProtocol[ObservationT, StateT, ActionT, ParamsT],
) -> GymnaxCompatibilityBridge[ObservationT, StateT, ActionT, ParamsT]:
    return GymnaxCompatibilityBridge(env)