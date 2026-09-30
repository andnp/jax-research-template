"""Declared worst-case substitution for diverged stored per-run scalars."""

from __future__ import annotations

import math

import polars as pl
import pytest
from research_analysis.dataset.types import CellAccounting, ExperimentDataset, ExperimentDimensions, RunAccounting
from research_analysis.non_finite import NonFiniteValuesError, substitute_non_finite
from research_analysis.performance import StoredValue, dataset_run_scores

_ACCOUNTING = RunAccounting(cells=(CellAccounting((("condition_name", "a"),), 2, 2, 0, 0, {"auc": 1}, {}),))


def _dataset() -> ExperimentDataset:
    runs = pl.DataFrame(
        {"condition_name": ["a", "a"], "seed": [0, 1], "run_id": [1, 2], "execution_id": [1, 1]}
    ).with_columns(ablation=pl.lit(None, pl.String), non_finite_metrics=pl.Series([[], ["auc", "loss"]]))
    curves = runs.select("condition_name", "seed", "run_id", "execution_id").with_columns(
        step=pl.lit(0), metric=pl.lit("reward"), value=pl.lit(1.0)
    )
    scalars = pl.concat(
        [
            runs.select("condition_name", "seed", "run_id", "execution_id", "ablation").with_columns(
                metric=pl.lit(metric), value=pl.Series([1.0, math.nan])
            )
            for metric in ("auc", "loss")
        ]
    )
    execution_metrics = pl.DataFrame({"execution_id": [1], "metric": ["auc"], "value": [math.nan]})
    return ExperimentDataset(
        "exp",
        ExperimentDimensions(),
        runs,
        curves,
        (),
        accounting=_ACCOUNTING,
        scalar_metrics=scalars,
        execution_metrics=execution_metrics,
    )


class TestSubstituteStoredScalars:
    def test_a_declared_worst_case_scores_the_diverged_stored_value(self) -> None:
        """A StoredValue analysis with a declared worst case scores instead of raising."""
        with pytest.raises(NonFiniteValuesError):
            dataset_run_scores(_dataset(), "auc", StoredValue())

        substituted = substitute_non_finite(_dataset(), "auc", -10.0)

        assert dataset_run_scores(substituted, "auc", StoredValue())["value"].to_list() == [1.0, -10.0]

    def test_only_the_named_run_scalar_is_rewritten(self) -> None:
        """Other scalar metrics stay loud, and whole-execution values are not per-run outcomes to substitute."""
        substituted = substitute_non_finite(_dataset(), "auc", -10.0)

        with pytest.raises(NonFiniteValuesError):
            dataset_run_scores(substituted, "loss", StoredValue())
        assert math.isnan(substituted.execution_metrics["value"][0])

    def test_divergence_records_survive_substitution(self) -> None:
        """The runs table and accounting still show that the substituted run diverged."""
        substituted = substitute_non_finite(_dataset(), "auc", -10.0)

        assert substituted.runs["non_finite_metrics"].to_list() == [[], ["auc", "loss"]]
        assert substituted.accounting == _ACCOUNTING
