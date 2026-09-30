"""Contract tests for restricting an experiment dataset to a subset of its runs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from research_analysis.dataset.types import (
    CellAccounting,
    ComponentVersionProvenance,
    ExecutionProvenance,
    ExperimentDataset,
    ExperimentDimensions,
    RunAccounting,
    RunArtifacts,
)
from research_analysis.diagnostics.types import DatasetDiagnostic, DiagnosticCode

# (condition, seed, run_id, execution_id); runs 1 and 2 share execution 10.
_RUNS = [("a", 0, 1, 10), ("b", 0, 2, 10), ("b", 1, 3, 11)]
_IDENTITY = ["condition_name", "seed", "run_id", "execution_id"]


def _dataset() -> ExperimentDataset:
    runs = pl.DataFrame(_RUNS, schema=_IDENTITY, orient="row").with_columns(pl.lit(None, pl.String).alias("ablation"))
    metrics = pl.DataFrame(
        [(*run, step, "reward", float(step)) for run in _RUNS for step in (0, 1)],
        schema=[*_IDENTITY, "step", "metric", "value"],
        orient="row",
    )
    # A stray scalar carries run 1's id under run 3's execution: it belongs to no loaded run.
    scalar_metrics = pl.DataFrame(
        [(*run, None, "auc", float(run[2])) for run in _RUNS] + [("a", 0, 1, 11, None, "auc", 99.0)],
        schema={
            "condition_name": pl.String,
            "seed": pl.Int64,
            "run_id": pl.Int64,
            "execution_id": pl.Int64,
            "ablation": pl.String,
            "metric": pl.String,
            "value": pl.Float64,
        },
        orient="row",
    )
    execution_metrics = pl.DataFrame(
        [(10, "throughput", 3.5), (11, "throughput", 4.5)], schema=["execution_id", "metric", "value"], orient="row"
    )
    return ExperimentDataset(
        "exp",
        ExperimentDimensions(),
        runs,
        metrics,
        (ComponentVersionProvenance(1, "ppo", "ALGO", 7, 1, "abc123"),),
        artifacts={
            (run_id, execution_id): RunArtifacts(Path(f"/runs/{run_id}"), None, {}) for _, _, run_id, execution_id in _RUNS
        },
        diagnostics=(DatasetDiagnostic(DiagnosticCode.RUN_FAILED, "run 4 failed", 4, 12),),
        accounting=RunAccounting(cells=(CellAccounting((("condition_name", "b"),), 3, 2, 1, 0, {}, {}),)),
        executions=tuple(ExecutionProvenance(execution_id, None, False, None, None, None) for execution_id in (10, 11)),
        scalar_metrics=scalar_metrics,
        execution_metrics=execution_metrics,
    )


class TestRunKeyedFrames:
    def test_keeps_only_the_selected_runs_curves_and_scalars(self) -> None:
        """Curve and run-scalar rows follow the selected runs; excluded runs contribute nothing."""
        subset = _dataset().filter_runs(pl.col("condition_name") == "b")

        assert subset.runs["run_id"].to_list() == [2, 3]
        assert subset.metrics.select("run_id", "step").rows() == [(2, 0), (2, 1), (3, 0), (3, 1)]
        assert subset.scalar_metrics.select("run_id", "value").rows() == [(2, 2.0), (3, 3.0)]

    def test_matches_scalars_on_run_and_execution_together(self) -> None:
        """A scalar whose run id survives under a different execution is not that run's value."""
        subset = _dataset().filter_runs(pl.col("run_id") == 1)

        assert subset.scalar_metrics.select("run_id", "execution_id", "value").rows() == [(1, 10, 1.0)]

    def test_never_duplicates_rows(self) -> None:
        """Keeping every run returns each frame row exactly once."""
        dataset = _dataset()

        subset = dataset.filter_runs(pl.lit(True))

        assert subset.metrics.equals(dataset.metrics)
        assert subset.scalar_metrics.height == len(_RUNS)
        assert subset.execution_metrics.equals(dataset.execution_metrics)

    def test_an_empty_selection_keeps_every_schema(self) -> None:
        """Selecting no run leaves empty frames that still satisfy the dataset's column contract."""
        dataset = _dataset()

        subset = dataset.filter_runs(pl.col("seed") > 5)

        assert subset.runs.is_empty() and subset.metrics.is_empty() and subset.scalar_metrics.is_empty()
        assert subset.execution_metrics.is_empty() and subset.artifacts == {}
        assert subset.metrics.schema == dataset.metrics.schema
        assert subset.scalar_metrics.schema == dataset.scalar_metrics.schema

    def test_artifacts_follow_their_runs(self) -> None:
        """Only the selected runs keep their artifact references."""
        subset = _dataset().filter_runs(pl.col("condition_name") == "b")

        assert sorted(subset.artifacts) == [(2, 10), (3, 11)]


class TestExecutionScopedValues:
    def test_a_shared_execution_keeps_its_metric_while_any_run_survives(self) -> None:
        """An execution metric is a whole-execution value, so one surviving run keeps it."""
        subset = _dataset().filter_runs(pl.col("run_id") == 2)

        assert subset.execution_metrics.rows() == [(10, "throughput", 3.5)]

    def test_an_execution_without_surviving_runs_drops_its_metric(self) -> None:
        """Once none of an execution's runs is selected, its metric no longer describes the subset."""
        subset = _dataset().filter_runs(pl.col("condition_name") == "a")

        assert subset.execution_metrics["execution_id"].to_list() == [10]


class TestSourceWideRecords:
    def test_diagnostics_accounting_and_executions_describe_the_whole_load(self) -> None:
        """Filtering is an analysis view; what was planned, run, and diagnosed is unchanged."""
        dataset = _dataset()

        subset = dataset.filter_runs(pl.col("condition_name") == "a")

        assert subset.diagnostics == dataset.diagnostics
        assert subset.accounting == dataset.accounting
        assert subset.executions == dataset.executions
        assert subset.provenance == dataset.provenance
