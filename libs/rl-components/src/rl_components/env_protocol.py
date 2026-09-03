"""Canonical single-environment protocol for shared RL components."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import chex
import jax
import jax.numpy as jnp

from rl_components.structs import chex_struct

TruncationPolicy = Literal["bootstrap", "terminate"]
"""How a task wants a time-limit truncation treated when bootstrapping.

``"bootstrap"`` marks a time-unlimited task, where the limit is a training device
external to the problem: a truncation keeps its bootstrap with coefficient ``gamma``.
``"terminate"`` marks a time-limited task, where the horizon is part of the MDP and the
optimal policy is time-dependent: a truncation kills the bootstrap like a real
termination. Either way the truncation breaks the trajectory.
"""


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
    truncation_policy: TruncationPolicy = "bootstrap"

    def __post_init__(self) -> None:
        if self.truncation_policy not in ("bootstrap", "terminate"):
            raise ValueError(f"truncation_policy must be 'bootstrap' or 'terminate', got {self.truncation_policy!r}")

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