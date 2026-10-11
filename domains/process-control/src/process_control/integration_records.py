from collections.abc import Mapping
from enum import IntEnum

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass

NO_REASON = 0


class AdvanceStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    INVALID_STAGE = 2
    RETRY_EXHAUSTED = 3
    EVENT_LOCATION_FAILED = 4
    DEPLETED = 5
    EVENT_BUDGET_EXHAUSTED = 6


@jax_dataclass
class CheckResult:
    """Fixed-order failure flags with the integer reason id of each; arrays so a failure never raises under trace."""

    failed: jax.Array
    reasons: jax.Array

    @property
    def any_failed(self) -> jax.Array:
        return jnp.any(self.failed, axis=-1)

    @property
    def first_reason(self) -> jax.Array:
        """Reason id of the first failing flag in declared order, or NO_REASON when none failed; batched over leading axes."""
        trailing = (*self.failed.shape[:-1], 1)
        flags = jnp.concatenate([self.failed, jnp.ones(trailing, dtype=bool)], axis=-1)
        reasons = jnp.concatenate([self.reasons, jnp.full(trailing, NO_REASON, dtype=self.reasons.dtype)], axis=-1)
        return jnp.take_along_axis(reasons, jnp.argmax(flags, axis=-1)[..., None], axis=-1)[..., 0]


def check_result(failed: tuple[jax.Array, ...]) -> CheckResult:
    """Pack failure flags; the reason id of flag i is i + 1."""
    flags = jnp.stack([jnp.asarray(flag, dtype=bool) for flag in failed]) if failed else jnp.zeros((0,), dtype=bool)
    return CheckResult(flags, jnp.arange(1, len(failed) + 1, dtype=jnp.int32))


def nonfinite_check(named: Mapping[str, object]) -> CheckResult:
    """Flag each named pytree that holds a non-finite leaf, in the mapping's order."""

    def has_nonfinite(tree: object) -> jax.Array:
        return jnp.any(jnp.stack([jnp.any(~jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)] or [jnp.array(False)]))

    return check_result(tuple(has_nonfinite(tree) for tree in named.values()))


@jax_dataclass
class IntegrationContext:
    """Persistent discrete modes, keyed `node.mode`, as integer or Boolean leaves held outside the integrated state."""

    modes: dict[str, jax.Array]


@jax_dataclass
class AdvanceChecks:
    """Why an advance stopped and how much bounded work it spent."""

    reason: jax.Array
    event_index: jax.Array
    retries: jax.Array
    segments: jax.Array
    location_iterations: jax.Array
    transitions: jax.Array


@jax_dataclass
class AdvanceResult:
    """Outcome of an advance: the last accepted state and how far it got, never a rejected trial."""

    state: Mapping[str, object]
    completed_time_h: jax.Array
    status: jax.Array
    quadratures: Mapping[str, object]
    context: IntegrationContext
    checks: AdvanceChecks
