"""Canonical single-environment protocol for shared RL components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chex
import jax
import jax.numpy as jnp

from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class EnvSpec:
    id: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    observation_dtype: jnp.dtype = jnp.float32
    action_dtype: jnp.dtype = jnp.int32
    num_actions: int | None = None


@chex_struct(frozen=True)
class EnvReset[ObservationT, StateT]:
    observation: ObservationT
    state: StateT


@chex_struct(frozen=True)
class EnvStep[ObservationT, StateT]:
    observation: ObservationT
    state: StateT
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict[str, jax.Array]


@runtime_checkable
class EnvProtocol[ObservationT, StateT, ActionT, ParamsT](Protocol):
    def spec(self, params: ParamsT | None = None) -> EnvSpec: ...

    def reset(self, key: chex.PRNGKey, params: ParamsT | None = None) -> EnvReset[ObservationT, StateT]: ...

    def step(self, key: chex.PRNGKey, state: StateT, action: ActionT, params: ParamsT | None = None) -> EnvStep[ObservationT, StateT]: ...