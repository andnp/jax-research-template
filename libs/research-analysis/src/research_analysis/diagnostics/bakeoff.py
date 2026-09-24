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
    SMALL_SAMPLE_COVERAGE = "small_sample_coverage"


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


@dataclass(frozen=True)
class IncompleteCell:
    """Seeds one condition lacks in one block, beside the outcome counts of that condition's planned runs."""

    condition: str
    block: str
    missing_seeds: tuple[int, ...]
    failed: int
    pending: int
    excluded: int


class IncompleteBlocksError(ValueError):
    """Raised when missingness='error' meets a seed some conditions observed and others did not."""

    def __init__(self, cells: tuple[IncompleteCell, ...]) -> None:
        self.cells = cells
        listed = "; ".join(
            f"{cell.condition!r} in block {cell.block!r} lacks seeds {list(cell.missing_seeds)} "
            f"({cell.failed} failed, {cell.pending} pending, {cell.excluded} excluded)"
            for cell in cells
        )
        super().__init__(
            f"incomplete blocks under missingness='error': {listed}. Finish or rerun those seeds, or declare "
            "missingness='complete_case' (compare only seeds every condition observed) or 'blockwise' (keep "
            "seeds observed by at least two conditions) in the BakeoffSpec."
        )
