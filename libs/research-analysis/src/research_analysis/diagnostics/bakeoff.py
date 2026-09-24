"""Structured statistical warnings for bakeoff-style reporting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class BakeoffWarningCode(StrEnum):
    """Structured bakeoff-report warning categories."""

    HYPERPARAMETER_AT_BOUNDARY = "hyperparameter_at_boundary"
    INSUFFICIENT_SEEDS = "insufficient_seeds"
    NON_NORMAL_DISTRIBUTION = "non_normal_distribution"
    FLATLINED_LEARNING_CURVE = "flatlined_learning_curve"
    ZERO_VARIANCE = "zero_variance"


@dataclass(frozen=True)
class BakeoffWarning:
    """One statistical caveat about a condition's data, never a fatal error.

    ``cell`` names the ``block_by`` coordinates the warning applies to (e.g.
    ``(("env_name", "cartpole"),)``), empty when the report has no blocking axes.
    """

    code: BakeoffWarningCode
    message: str
    condition: str | None = None
    cell: tuple[tuple[str, object], ...] = ()


@dataclass
class BakeoffWarningAccumulator:
    """Collects BakeoffWarning entries in place, avoiding a return-value threading convention."""

    _items: list[BakeoffWarning] = field(default_factory=list)

    def add(
        self,
        code: BakeoffWarningCode,
        message: str,
        condition: str | None = None,
        cell: tuple[tuple[str, object], ...] = (),
    ) -> None:
        self._items.append(BakeoffWarning(code, message, condition, cell))

    def extend(self, warnings: Iterable[BakeoffWarning]) -> None:
        self._items.extend(warnings)

    def as_tuple(self) -> tuple[BakeoffWarning, ...]:
        return tuple(self._items)
