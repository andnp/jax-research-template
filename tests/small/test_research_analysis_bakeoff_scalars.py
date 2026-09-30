"""Bakeoff contracts for comparisons scored from stored per-run scalars."""

from __future__ import annotations

import math

import polars as pl
import pytest
from research_analysis.dataset.types import ExperimentDataset, ExperimentDimensions
from research_analysis.non_finite import NonFiniteValuesError
from research_analysis.performance import StoredValue
from research_analysis.reporting import bakeoff_report

_SEEDS = 5


def _scalar_dataset(scores: dict[str, list[float]]) -> ExperimentDataset:
    rows = [
        (condition, seed, index * 100 + seed, value)
        for index, (condition, values) in enumerate(scores.items())
        for seed, value in enumerate(values)
    ]
    identity = pl.DataFrame(rows, schema=["condition_name", "seed", "run_id", "value"], orient="row").with_columns(
        execution_id=pl.col("run_id"), ablation=pl.lit(None, pl.String)
    )
    runs = identity.drop("value")
    curves = runs.with_columns(step=pl.lit(0), metric=pl.lit("reward"), value=pl.lit(0.0))
    scalars = identity.with_columns(metric=pl.lit("auc"))
    return ExperimentDataset("demo", ExperimentDimensions(), runs, curves, (), scalar_metrics=scalars)


def _scores(offset: float) -> list[float]:
    return [offset + seed for seed in range(_SEEDS)]


class TestExcludedRunsStayOut:
    def test_a_diverged_scalar_outside_the_compared_conditions_is_ignored(self) -> None:
        """Only the compared conditions' stored values are scored; an excluded arm's NaN cannot abort the report."""
        dataset = _scalar_dataset({"a": _scores(0.0), "b": _scores(1.0), "c": [math.nan] * _SEEDS})

        report = bakeoff_report(dataset, "auc", ["a", "b"], measure=StoredValue())

        assert sorted(report.cells) == ["a", "b"]
        assert {run.condition for run in report.runs} == {"a", "b"}

    def test_a_diverged_scalar_in_a_compared_condition_still_raises(self) -> None:
        """Divergence inside the comparison is an outcome and must fail loudly without a declared worst case."""
        dataset = _scalar_dataset({"a": _scores(0.0), "b": [*_scores(1.0)[:-1], math.nan]})

        with pytest.raises(NonFiniteValuesError):
            bakeoff_report(dataset, "auc", ["a", "b"], measure=StoredValue())
