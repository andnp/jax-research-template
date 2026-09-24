"""Contract tests for report documents: their JSON form and their rendering."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import (
    CellAccounting,
    ExclusionReason,
    ExecutionProvenance,
    ExperimentDataset,
    ExperimentDimensions,
    RunAccounting,
)
from research_analysis.diagnostics.types import DatasetDiagnostic, DiagnosticCode
from research_analysis.performance import TailMean
from research_analysis.plan import AnalysisPlan, BakeoffAnalysis
from research_analysis.report import BakeoffResult, PlanStatus, ReportStatus, exploratory_report
from research_analysis.report_json import plan_fingerprint, report_to_json
from research_analysis.reporting import bakeoff_report


def _dataset(scores: dict[str, list[float]], diagnostics: tuple[DatasetDiagnostic, ...] = ()) -> ExperimentDataset:
    rows = [
        (condition, seed, index * 100 + seed, value)
        for index, (condition, values) in enumerate(scores.items())
        for seed, value in enumerate(values)
    ]
    runs = pl.DataFrame(
        {
            "condition_name": [condition for condition, _seed, _run, _value in rows],
            "seed": [seed for _condition, seed, _run, _value in rows],
            "run_id": [run for _condition, _seed, run, _value in rows],
            "execution_id": [run for _condition, _seed, run, _value in rows],
            "ablation": pl.Series([None] * len(rows), dtype=pl.String),
        }
    )
    metrics = runs.select("condition_name", "seed", "run_id", "execution_id").with_columns(
        step=pl.lit(0), metric=pl.lit("return"), value=pl.Series([value for *_rest, value in rows])
    )
    accounting = RunAccounting(
        cells=tuple(
            CellAccounting(
                cell=(("condition_name", condition),),
                planned=len(values) + 1,
                completed=len(values),
                failed=1,
                pending=0,
                diverged={},
                excluded={},
            )
            for condition, values in scores.items()
        ),
        unassigned={ExclusionReason.MISSING_HYPER_CONFIG: 2},
    )
    return ExperimentDataset(
        experiment_name="demo",
        dimensions=ExperimentDimensions(),
        runs=runs,
        metrics=metrics,
        provenance=(),
        diagnostics=diagnostics,
        accounting=accounting,
        executions=(ExecutionProvenance(0, "abc123", True, "node-1", None, None),),
    )


_SCORES = {"base": [1.0, 2.0, 3.0, 4.0], "new": [2.0, 3.5, 4.0, 6.0]}


def _bakeoff_document(dataset: ExperimentDataset) -> dict[str, Any]:
    report = bakeoff_report(
        dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50), verbose=False, measure=TailMean(0.5)
    )
    return json.loads(report_to_json(exploratory_report(dataset, Path("experiments.sqlite"), [BakeoffResult("headline", report)])))


class TestReportJson:
    def test_document_names_its_schema_and_labels_hand_chosen_analyses_exploratory(self) -> None:
        """Every document is versioned, and one built from ad-hoc analyses says so with a reason."""
        document = _bakeoff_document(_dataset(_SCORES))

        assert document["schema_version"] == 1
        assert document["status"] == ReportStatus.OK
        assert document["plan_status"] == PlanStatus.EXPLORATORY
        assert document["exploratory_reasons"]
        assert document["plan"] is None

    def test_accounting_cells_and_execution_git_state_are_readable_objects(self) -> None:
        """Cells are keyed by axis name, and each execution states whether it ran uncommitted code."""
        document = _bakeoff_document(_dataset(_SCORES))

        accounting = document["accounting"]
        assert accounting["cells"][0]["cell"] == {"condition_name": "base"}
        assert accounting["cells"][0]["failed"] == 1
        assert accounting["unassigned"] == {"missing_hyper_config": 2}
        assert document["provenance"]["executions"] == [
            {"execution_id": 0, "git_commit": "abc123", "dirty": True, "hostname": "node-1", "start_time": None, "end_time": None}
        ]

    def test_results_carry_their_kind_measure_and_every_run(self) -> None:
        """A reader can tell which analysis and measure produced the numbers, and recheck them run by run."""
        document = _bakeoff_document(_dataset(_SCORES))

        result = document["analyses"][0]
        assert (result["kind"], result["name"]) == ("bakeoff", "headline")
        assert result["report"]["measure"] == {"kind": "tail_mean", "fraction": 0.5}
        assert len(result["report"]["runs"]) == 8

    def test_fatal_diagnostics_withhold_every_result(self) -> None:
        """A load with a fatal diagnostic yields its diagnostics and accounting, never a partial result."""
        fatal = DatasetDiagnostic(DiagnosticCode.MALFORMED_ARTIFACT, "run 7 has no metrics database", 7)
        pending = DatasetDiagnostic(DiagnosticCode.RUN_PENDING, "run 8 has not completed yet", 8)

        document = _bakeoff_document(_dataset(_SCORES, (fatal, pending)))

        assert document["status"] == ReportStatus.LOAD_FAILED
        assert document["analyses"] == []
        assert [item["code"] for item in document["fatal_diagnostics"]] == ["malformed_artifact"]
        assert [item["code"] for item in document["non_fatal_diagnostics"]] == ["run_pending"]

    def test_a_non_finite_number_fails_serialization(self) -> None:
        """Strict JSON: a NaN reaching a report is an error, not an unparseable file."""
        dataset = _dataset(_SCORES)
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50), verbose=False)
        broken = replace(report, comparison_confidence=math.nan)

        with pytest.raises(ValueError, match="JSON compliant"):
            report_to_json(exploratory_report(dataset, Path("db"), [BakeoffResult("headline", broken)]))


class TestPlanFingerprint:
    def test_identical_declarations_share_a_fingerprint(self) -> None:
        """Two separately built but equal plans hash the same, so a snapshot can be matched later."""
        first = AnalysisPlan("return", (BakeoffAnalysis("h", ("a", "b")),), ExperimentDimensions())
        second = AnalysisPlan("return", (BakeoffAnalysis("h", ("a", "b")),), ExperimentDimensions())

        assert plan_fingerprint(first) == plan_fingerprint(second)

    def test_any_change_to_the_plan_changes_its_fingerprint(self) -> None:
        """Changing the measure after the fact must be detectable."""
        plan = AnalysisPlan("return", (BakeoffAnalysis("h", ("a", "b")),), ExperimentDimensions())

        assert plan_fingerprint(plan) != plan_fingerprint(replace(plan, measure=TailMean(0.1)))
