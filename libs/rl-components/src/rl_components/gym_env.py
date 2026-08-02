"""Gymnax-style environment Protocol for RL agent typing.

Agents built on the Gymnax compatibility layer accept the tuple-returning
interface described here.  New-style environments implementing
``EnvProtocol`` should be wrapped with ``make_gymnax_compat_env`` before
being passed to any ``make_train`` function.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp


class ObservationSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> jnp.dtype: ...


class DiscreteActionSpace(Protocol):
    @property
    def n(self) -> int: ...


class ContinuousActionSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class BoundedContinuousActionSpace(ContinuousActionSpace, Protocol):
    @property
    def action_low(self) -> jax.Array | None: ...

    @property
    def action_high(self) -> jax.Array | None: ...


class GymEnv[ActionSpaceT](Protocol):
    def observation_space(self, params: object | None = None) -> ObservationSpace: ...

    def action_space(self, params: object | None = None) -> ActionSpaceT: ...

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, object]: ...

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, dict[str, jax.Array]]: ...
