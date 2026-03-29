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
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None

    def __post_init__(self) -> None:
        action_dtype = jnp.dtype(self.action_dtype)

        if self.num_actions is not None:
            if self.num_actions <= 0:
                raise ValueError(f"num_actions must be positive, got {self.num_actions}")
            if self.action_shape != ():
                raise ValueError(f"discrete action spaces must use scalar action_shape=(), got {self.action_shape}")
            if not jnp.issubdtype(action_dtype, jnp.integer):
                raise TypeError(f"discrete action spaces must use an integer action_dtype, got {action_dtype}")
            if self.action_low is not None or self.action_high is not None:
                raise ValueError("discrete action spaces cannot declare continuous action bounds")
            return

        if not jnp.issubdtype(action_dtype, jnp.floating):
            raise TypeError(f"continuous action spaces must use a floating-point action_dtype, got {action_dtype}")
        if (self.action_low is None) != (self.action_high is None):
            raise ValueError("continuous action bounds require both action_low and action_high")


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