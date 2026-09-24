"""Structured statistical warnings for bakeoff-style reporting."""

from __future__ import annotations

from dataclasses import dataclass
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
