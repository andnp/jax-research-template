"""Declared per-run performance measures that reduce a logged curve to one scalar."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

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


type PerformanceMeasure = FinalValue | TailMean | ReturnRate


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


def run_scores(frame: pl.DataFrame, metric: str, measure: PerformanceMeasure) -> pl.DataFrame:
    """Reduce each run's ``metric`` curve to one ``value``, keeping the frame's run identity columns."""
    keys = [column for column in _IDENTITY_COLUMNS if column in frame.columns]
    return (
        frame.filter(pl.col("metric") == metric)
        .group_by(keys)
        .agg(_score(measure).alias("value"))
        .sort(keys)
    )
