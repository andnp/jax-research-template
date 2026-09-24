"""Contract tests for report documents: their JSON form and their rendering."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from research_analysis._reporting_warnings import small_sample_coverage_warning
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
from research_analysis.report import (
    AnalysisResult,
    BakeoffResult,
    BestHypersResult,
    PlanStatus,
    ReportStatus,
    exploratory_report,
)
from research_analysis.report_json import plan_fingerprint, report_to_json
from research_analysis.report_markdown import render_markdown, render_summary
from research_analysis.report_plots import write_report_plots
from research_analysis.reporting import HyperparameterConfigReport, bakeoff_report


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
        dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50), measure=TailMean(0.5)
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
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50))
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


def _markdown(dataset: ExperimentDataset, results: list[AnalysisResult]) -> str:
    return render_markdown(exploratory_report(dataset, Path("experiments.sqlite"), results))


class TestReportMarkdown:
    def test_every_interval_states_its_confidence_and_no_spread_is_written_as_plus_minus(self) -> None:
        """Honest display: CIs carry their level, family CIs their adjusted level, and nothing reads 'mean ± x'."""
        dataset = _dataset({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 3.5, 6.0], "c": [0.0, 1.0, 1.5, 2.0]})
        report = bakeoff_report(dataset, "return", ["a", "b", "c"], spec=BakeoffSpec(n_resamples=50))

        text = _markdown(dataset, [BakeoffResult("headline", report)])

        assert "±" not in text
        assert "mean with its 95% CI" in text
        assert f"each CI at {report.comparison_confidence * 100:.4g}%" in text
        assert "Holm-adjusted" in text

    def test_skipped_cells_and_pairs_state_why(self) -> None:
        """A cell or pair below min_samples appears with its reason instead of silently vanishing."""
        dataset = _dataset({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0], "c": [5.0, 6.0]})
        report = bakeoff_report(
            dataset, "return", ["a", "b", "c"], spec=BakeoffSpec(missingness="blockwise", n_resamples=50)
        )

        text = _markdown(dataset, [BakeoffResult("headline", report)])

        assert "none: only 2 runs, fewer than min_samples=3" in text
        assert "skipped: only 2 paired runs" in text

    def test_accounting_and_diagnostics_precede_results(self) -> None:
        """A reader meets failed and pending runs before any number built on the survivors."""
        pending = DatasetDiagnostic(DiagnosticCode.RUN_PENDING, "run 8 has not completed yet", 8)
        dataset = _dataset(_SCORES, (pending,))
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50))

        text = _markdown(dataset, [BakeoffResult("headline", report)])

        assert text.index("## Run accounting") < text.index("## Diagnostics") < text.index("## headline")
        assert "run 8 has not completed yet" in text
        assert "**exploratory**" in text

    def test_a_fatal_load_renders_diagnostics_and_no_results(self) -> None:
        """With a fatal diagnostic the rendering says no results are reported and shows none."""
        fatal = DatasetDiagnostic(DiagnosticCode.MALFORMED_ARTIFACT, "run 7 has no metrics database", 7)
        dataset = _dataset(_SCORES, (fatal,))
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50))

        text = _markdown(dataset, [BakeoffResult("headline", report)])

        assert "**Fatal: no results are reported.**" in text
        assert "## headline" not in text

    def test_a_champion_is_reported_by_its_corrected_mean_beside_its_bias(self) -> None:
        """The raw argmax mean is shown only next to the maximization bias it carries."""
        champion = HyperparameterConfigReport(
            winning_config={"lr": 0.1},
            raw_mean=5.0,
            corrected_mean=4.2,
            maximization_bias=0.8,
            estimate_ci=(3.9, 4.6),
            unpaired_run_ids=(3, 4),
        )

        text = _markdown(_dataset(_SCORES), [BestHypersResult("tuned", {"algorithm": "ppo"}, champion)])

        assert "Champion: lr=0.1" in text
        assert "Corrected mean 4.2, 95% CI [3.9, 4.6]" in text
        assert "maximization bias 0.8" in text
        assert "left out: 2" in text

    def test_a_champion_warning_is_rendered_and_serialized(self) -> None:
        """A caveat on a champion interval reaches both the markdown reader and the JSON consumer."""
        warning = small_sample_coverage_warning(5, "champion")
        assert warning is not None
        champion = HyperparameterConfigReport(
            winning_config={"lr": 0.1},
            raw_mean=5.0,
            corrected_mean=4.2,
            maximization_bias=0.8,
            estimate_ci=(3.9, 4.6),
            warnings=(warning,),
        )
        results: list[AnalysisResult] = [BestHypersResult("tuned", {}, champion)]

        text = _markdown(_dataset(_SCORES), results)
        payload = json.loads(report_to_json(exploratory_report(_dataset(_SCORES), Path("db"), results)))

        assert "`small_sample_coverage` (all) : the champion CI rests on 5 runs" in text
        assert [w["code"] for w in payload["analyses"][0]["report"]["warnings"]] == ["small_sample_coverage"]


    def test_summary_gives_status_run_outcomes_and_each_headline_difference(self) -> None:
        """The console summary is short but still states run outcomes and each difference with its confidence."""
        dataset = _dataset(_SCORES)
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50))
        document = exploratory_report(dataset, Path("db"), [BakeoffResult("headline", report)])

        summary = render_summary(document)

        difference = report.difference("base", "new")
        assert "- Status: **ok**" in summary
        assert "8 completed, 2 failed" in summary
        assert f"base − new: {difference.estimate:.4g}" in summary
        assert "at 95% (n=4)" in summary

class TestReportPlots:
    def test_figures_are_written_under_the_report_directory_and_listed_relative_to_it(self, tmp_path: Path) -> None:
        """A report directory is self-contained: the document names its figures by relative path."""
        dataset = _dataset(_SCORES)
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50))
        document = exploratory_report(dataset, Path("db"), [BakeoffResult("headline", report)])

        written = write_report_plots(document, tmp_path)

        assert list(written.plots.values()) == [Path("headline__run_distribution.png")]
        assert all((tmp_path / path).exists() for path in written.plots.values())
        assert "![headline__run_distribution](headline__run_distribution.png)" in render_markdown(written)
