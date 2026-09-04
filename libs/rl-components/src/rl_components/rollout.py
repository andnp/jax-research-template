"""Generic JAX mechanics for batched stateful-controller rollouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple, Protocol

import chex
import jax

from rl_components.env_protocol import EnvReset, EnvStep

ResetPolicy = Literal["ignore", "auto_reset"]
BatchedRNGMode = Literal["two-way", "three-way"]

type Telemetry = Mapping[str, jax.Array]


class BatchedEnvironment[ObservationT, EnvStateT, ActionT](Protocol):
    """Environment methods required by the batched rollout kernel."""

    def reset(self, key: chex.PRNGKey) -> EnvReset[ObservationT, EnvStateT]: ...

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvStateT,
        action: ActionT,
    ) -> EnvStep[ObservationT, EnvStateT]: ...


class ActionStep[ObservationT, ControllerStateT, ActionT](Protocol):
    """Stateful action callback used by one scan iteration."""

    def __call__(
        self,
        observations: ObservationT,
        controller_state: ControllerStateT,
        previous_action: jax.Array,
        step_index: jax.Array,
        action_key: jax.Array | None,
    ) -> tuple[ActionT, ControllerStateT]: ...


class TelemetryHook(Protocol):
    """Callback for adding static-key traced telemetry."""

    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        previous_action: jax.Array,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        info: Telemetry,
    ) -> Telemetry: ...


class BatchedRolloutCarry[EnvStateT, ControllerStateT](NamedTuple):
    """JAX-compatible state for a seed-batched rollout."""

    observations: jax.Array
    environment_state: EnvStateT
    controller_state: ControllerStateT
    rng: jax.Array
    previous_action: jax.Array


class BatchedRolloutResult[EnvStateT, ControllerStateT](NamedTuple):
    """Final carry, generic metrics, boundaries, and environment diagnostics."""

    final_carry: BatchedRolloutCarry[EnvStateT, ControllerStateT]
    metrics: dict[str, jax.Array]
    terminated: jax.Array
    truncated: jax.Array
    info: dict[str, jax.Array]

    @property
    def done(self) -> jax.Array:
        """Return the combined environment boundary signal."""
        return self.terminated | self.truncated


ActionStepFn = Callable[
    [jax.Array, object, jax.Array, jax.Array, jax.Array | None],
    tuple[jax.Array, object],
]


__all__ = [
    "ActionStep",
    "ActionStepFn",
    "BatchedEnvironment",
    "BatchedRNGMode",
    "BatchedRolloutCarry",
    "BatchedRolloutResult",
    "ResetPolicy",
    "Telemetry",
    "TelemetryHook",
]
