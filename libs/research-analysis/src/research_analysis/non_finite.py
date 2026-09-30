"""Fail loudly on diverged (NaN or infinite) run values unless a policy is declared explicitly."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import polars as pl
from numpy.typing import NDArray

from research_analysis.dataset.types import ExperimentDataset

_IDENTITY_COLUMNS = ("condition_name", "env_name", "run_id", "seed")
_LISTED_RUNS = 10


class NonFiniteValuesError(ValueError):
    """Raised when a statistic would consume NaN or infinite run values."""

    def __init__(self, context: str, offenders: pl.DataFrame) -> None:
        self.offenders = offenders
        listed = offenders.head(_LISTED_RUNS).to_dicts()
        more = f" and {offenders.height - _LISTED_RUNS} more" if offenders.height > _LISTED_RUNS else ""
        super().__init__(
            f"{context} has non-finite values for {listed}{more}; diverged runs are outcomes, not missing data. "
            "Declare a worst-case value with research_analysis.non_finite.substitute_non_finite or analyse "
            "divergence separately."
        )


def require_finite(frame: pl.DataFrame, context: str) -> None:
    offenders = frame.filter(~pl.col("value").is_finite())
    if offenders.is_empty():
        return
    columns = [column for column in _IDENTITY_COLUMNS if column in frame.columns]
    raise NonFiniteValuesError(context, offenders.select(columns).unique(maintain_order=True))


def require_finite_array(values: NDArray[np.floating], name: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if finite.all():
        return
    rows = np.flatnonzero(~finite.reshape(finite.shape[0], -1).all(axis=1)).tolist() if array.ndim else [0]
    raise ValueError(
        f"{name} has non-finite values in rows {rows}; diverged runs are outcomes, not missing data. Declare a "
        "worst-case value with research_analysis.non_finite.substitute_non_finite or analyse divergence separately."
    )


def substitute_non_finite(dataset: ExperimentDataset, metric: str, worst_case: float) -> ExperimentDataset:
    """Replace one metric's non-finite curve and run-scalar values with a declared worst case, as an analysis choice.

    The runs table keeps its ``non_finite_metrics`` flags and the accounting keeps its diverged counts, so the
    substitution never hides that the runs diverged.
    """
    if not math.isfinite(worst_case):
        raise ValueError(f"worst_case must be finite, got {worst_case!r}")
    diverged = (pl.col("metric") == metric) & ~pl.col("value").is_finite()
    substituted = pl.when(diverged).then(pl.lit(worst_case)).otherwise(pl.col("value")).alias("value")
    return replace(
        dataset,
        metrics=dataset.metrics.with_columns(substituted),
        scalar_metrics=dataset.scalar_metrics.with_columns(substituted),
    )
