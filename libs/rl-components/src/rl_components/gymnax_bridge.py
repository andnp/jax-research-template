"""Minimal Gymnax compatibility bridge for canonical environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

from rl_components.env_protocol import EnvProtocol, EnvSpec


@chex.dataclass(frozen=True)
class GymnaxSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype

    if TYPE_CHECKING:
        def __init__(
            self,
            *,
            shape: tuple[int, ...],
            dtype: jnp.dtype,
        ) -> None: ...


@chex.dataclass(frozen=True)
class GymnaxDiscreteSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    n: int

    if TYPE_CHECKING:
        def __init__(
            self,
            *,
            shape: tuple[int, ...],
            dtype: jnp.dtype,
            n: int,
        ) -> None: ...


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
    )


class GymnaxCompatibilityBridge[ObservationT, StateT, ActionT, ParamsT]:
    def __init__(self, env: EnvProtocol[ObservationT, StateT, ActionT, ParamsT]) -> None:
        self._env = env

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)

    def observation_space(self, params: ParamsT | None = None) -> GymnaxSpace:
        return _observation_space_from_spec(self._env.spec(params))

    def action_space(self, params: ParamsT | None = None) -> GymnaxSpace | GymnaxDiscreteSpace:
        return _action_space_from_spec(self._env.spec(params))

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None) -> tuple[ObservationT, StateT]:
        reset = self._env.reset(key, params)
        return reset.observation, reset.state

    def step(
        self,
        key: chex.PRNGKey,
        state: StateT,
        action: ActionT,
        params: ParamsT | None = None,
    ) -> tuple[ObservationT, StateT, jax.Array, jax.Array, dict[str, jax.Array]]:
        transition = self._env.step(key, state, action, params)
        done = jnp.logical_or(transition.terminated, transition.truncated)
        info = dict(transition.info)
        info.setdefault("terminated", jnp.asarray(transition.terminated))
        info.setdefault("truncated", jnp.asarray(transition.truncated))
        return transition.observation, transition.state, transition.reward, done, info


def make_gymnax_compat_env[ObservationT, StateT, ActionT, ParamsT](
    env: EnvProtocol[ObservationT, StateT, ActionT, ParamsT],
) -> GymnaxCompatibilityBridge[ObservationT, StateT, ActionT, ParamsT]:
    return GymnaxCompatibilityBridge(env)