"""Declared per-run performance measures for logged curves and stored scalar values."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from research_analysis.duplicates import duplicate_seed_error
from research_analysis.non_finite import require_finite

_IDENTITY_COLUMNS = ("condition_name", "env_name", "seed", "run_id", "execution_id", "ablation")


@dataclass(frozen=True)
class FinalValue:
    """The last logged value of each run, ordered by step."""


@dataclass(frozen=True)
class TailMean:
    """Mean of each run's values over the final ``fraction`` of its logged step span.

    Per-episode return metrics should be step-weighted before they are reduced.
    """

    fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")


@dataclass(frozen=True)
class ReturnRate:
    """Mean over every logged point of each run's curve, the handbook's return rate.

    Per-episode return metrics should be step-weighted before they are reduced.
    """


@dataclass(frozen=True)
class StoredValue:
    """Use a scalar already stored as the run's outcome, without curve reduction."""


type PerformanceMeasure = FinalValue | TailMean | ReturnRate | StoredValue

FINAL_VALUE = FinalValue()


def _score(measure: PerformanceMeasure) -> pl.Expr:
    match measure:
        case FinalValue():
            return pl.col("value").sort_by("step").last()
        case TailMean(fraction=fraction):
            step = pl.col("step")
            cutoff = step.min() + (1.0 - fraction) * (step.max() - step.min())
            return pl.col("value").filter(step >= cutoff).mean()
        case ReturnRate():
            return pl.col("value").mean()
        case StoredValue():
            raise ValueError("StoredValue requires a scalar metric frame without a 'step' column")


def run_scores(frame: pl.DataFrame, metric: str, measure: PerformanceMeasure) -> pl.DataFrame:
    """Score each run's ``metric`` rows, keeping its identity columns.

    A NaN or infinite score raises: a diverged run is an outcome, so declare a worst case with ``substitute_non_finite``.
    """
    keys = [column for column in _IDENTITY_COLUMNS if column in frame.columns]
    curves = frame.filter(pl.col("metric") == metric)
    is_curve = "step" in curves.columns
    if is_curve and isinstance(measure, StoredValue):
        raise ValueError("StoredValue requires a scalar metric frame without a 'step' column")
    if not is_curve and not isinstance(measure, StoredValue):
        raise ValueError("a scalar metric frame without a 'step' column requires StoredValue()")
    duplicate_keys = [*keys, "step"] if is_curve else keys
    pooled = curves.filter(pl.struct(duplicate_keys).is_duplicated())
    if not pooled.is_empty():
        first = pooled.row(0, named=True)
        if is_curve:
            raise duplicate_seed_error(f"metric {metric!r} for {first.get('condition_name')!r}", first.get("seed"))
        identity = {key: first.get(key) for key in keys}
        raise ValueError(f"metric {metric!r} has duplicate stored values for run identity {identity!r}")
    if is_curve:
        scores = curves.group_by(keys).agg(_score(measure).alias("value")).sort(keys)
    else:
        scores = curves.select([*keys, "value"]).sort(keys)
    require_finite(scores, f"{metric!r} run scores")
    return scores
