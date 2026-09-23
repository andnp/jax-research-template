"""Declared per-run performance measures that reduce a logged curve to one scalar."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_IDENTITY_COLUMNS = ("condition_name", "env_name", "seed", "run_id", "execution_id", "ablation")


@dataclass(frozen=True)
class FinalValue:
    """The last logged value of each run, ordered by step."""


type PerformanceMeasure = FinalValue


def _score(measure: PerformanceMeasure) -> pl.Expr:
    match measure:
        case FinalValue():
            return pl.col("value").sort_by("step").last()


def run_scores(frame: pl.DataFrame, metric: str, measure: PerformanceMeasure) -> pl.DataFrame:
    """Reduce each run's ``metric`` curve to one ``value``, keeping the frame's run identity columns."""
    keys = [column for column in _IDENTITY_COLUMNS if column in frame.columns]
    return (
        frame.filter(pl.col("metric") == metric)
        .group_by(keys)
        .agg(_score(measure).alias("value"))
        .sort(keys)
    )
