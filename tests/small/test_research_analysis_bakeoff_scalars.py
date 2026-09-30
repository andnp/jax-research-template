"""Bakeoff contracts for comparisons scored from stored per-run scalars."""

from __future__ import annotations

import math

import polars as pl
import pytest
from research_analysis.dataset.types import ExperimentDataset, ExperimentDimensions, SweepAxis, SweepValueType
from research_analysis.non_finite import NonFiniteValuesError
from research_analysis.performance import StoredValue
from research_analysis.reporting import bakeoff_report

_SEEDS = 5
# lr=0.1 wins for both arms.
_LRS = (0.1, 0.2)


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


def _tuned_dataset(weak_b_value: float | None = None) -> ExperimentDataset:
    rows = [
        (condition, seed, len(_LRS) * _SEEDS * arm + _SEEDS * index + seed, lr, value)
        for arm, condition in enumerate(("a", "b"))
        for index, lr in enumerate(_LRS)
        for seed, value in enumerate(_scores(arm + (10.0 if lr == 0.1 else 0.0)))
    ]
    if weak_b_value is not None:
        rows = [(*row[:4], weak_b_value if row[0] == "b" and row[3] == 0.2 else row[4]) for row in rows]
    identity = pl.DataFrame(rows, schema=["condition_name", "seed", "run_id", "lr", "value"], orient="row").with_columns(
        execution_id=pl.col("run_id"), ablation=pl.lit(None, pl.String)
    )
    runs = identity.drop("value")
    curves = runs.select("condition_name", "seed", "run_id", "execution_id").with_columns(
        step=pl.lit(0), metric=pl.lit("reward"), value=pl.lit(0.0)
    )
    scalars = identity.drop("lr").with_columns(metric=pl.lit("auc"))
    dimensions = ExperimentDimensions(sweep_axes=(SweepAxis("lr", SweepValueType.FLOAT),))
    return ExperimentDataset("demo", dimensions, runs, curves, (), scalar_metrics=scalars)


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


class TestTunedChampions:
    def test_only_champion_configurations_are_scored(self) -> None:
        """After selection, each arm's comparison uses its champion configuration's stored values alone."""
        dataset = _tuned_dataset()
        champion_runs = set(dataset.runs.filter(pl.col("lr") == 0.1)["run_id"].to_list())

        report = bakeoff_report(dataset, "auc", ["a", "b"], select_champion_over=["lr"], measure=StoredValue())

        assert report.champions is not None
        assert {arm: champion.winning_config for arm, champion in report.champions[""].items()} == {
            "a": {"lr": 0.1},
            "b": {"lr": 0.1},
        }
        assert {run.run_id for run in report.runs} == champion_runs

    def test_a_diverged_non_champion_configuration_blocks_selection(self) -> None:
        """Ranking needs every configuration's score, so a diverged one fails loudly rather than losing silently."""
        with pytest.raises(NonFiniteValuesError, match="'b'"):
            bakeoff_report(
                _tuned_dataset(weak_b_value=math.nan), "auc", ["a", "b"], select_champion_over=["lr"], measure=StoredValue()
            )
