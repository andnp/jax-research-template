"""Contract tests for report documents: their JSON form and their rendering."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from importlib.metadata import version
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
    SweepAxis,
    SweepValueType,
)
from research_analysis.declared_plan import PlanInterpretationError, plan_fingerprint, recorded_plan_fingerprint
from research_analysis.diagnostics.types import DatasetDiagnostic, DiagnosticCode
from research_analysis.performance import FinalValue, ReturnRate, StoredValue, TailMean
from research_analysis.plan import (
    AnalysisPlan,
    BakeoffAnalysis,
    BestHypersAnalysis,
    HypersDifferenceAnalysis,
    SliceHypersAnalysis,
    against_baseline,
)
from research_analysis.report import (
    AnalysisResult,
    BakeoffResult,
    BestHypersResult,
    HypersDifferenceResult,
    PlanStatus,
    ReportStatus,
    exploratory_report,
)
from research_analysis.report_files import report_directory, write_report_document
from research_analysis.report_json import JsonValue, report_to_json, to_jsonable
from research_analysis.report_markdown import render_markdown, render_summary
from research_analysis.report_plots import write_report_plots
from research_analysis.reporting import HyperparameterConfigReport, HyperparameterDifferenceReport, bakeoff_report


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

    def test_the_analysis_code_that_produced_the_report_is_recorded(self) -> None:
        """A changed default never changes a plan's status, so the report names the analysis version and commit."""
        document = _bakeoff_document(_dataset(_SCORES))

        toolchain = document["provenance"]["toolchain"]
        assert toolchain["package_version"] == version("research-analysis")
        assert re.fullmatch(r"[0-9a-f]{40}", toolchain["git_commit"])
        assert isinstance(toolchain["dirty"], bool)

    def test_results_carry_their_kind_measure_and_every_run(self) -> None:
        """A reader can tell which analysis and measure produced the numbers, and recheck them run by run."""
        document = _bakeoff_document(_dataset(_SCORES))

        result = document["analyses"][0]
        assert (result["kind"], result["name"]) == ("bakeoff", "headline")
        assert result["report"]["measure"] == {"kind": "tail_mean", "fraction": 0.5}
        assert len(result["report"]["runs"]) == 8

    def test_an_expanded_interval_is_recorded_and_widens_every_ci(self) -> None:
        """The JSON names the interval method in the spec and each difference; on the same resamples it only widens."""
        dataset = _dataset(_SCORES)
        results = {
            interval: bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50, interval=interval))
            for interval in ("percentile", "expanded_percentile")
        }
        document = json.loads(
            report_to_json(exploratory_report(dataset, Path("db"), [BakeoffResult("x", results["expanded_percentile"])]))
        )

        report = document["analyses"][0]["report"]
        assert report["spec"]["interval"] == "expanded_percentile"
        assert report["differences"][0]["method"] == "paired expanded percentile bootstrap"
        plain = results["percentile"].difference("new", "base").ci
        expanded = results["expanded_percentile"].difference("new", "base").ci
        assert plain is not None and expanded is not None
        assert expanded[0] <= plain[0] <= plain[1] <= expanded[1]

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


_BARE_PLAN = AnalysisPlan("return", (BakeoffAnalysis("h", ("a", "b")),), ExperimentDimensions())


def _recorded_parent(path: tuple[str | int, ...]) -> tuple[Any, str | int, Any]:
    recorded = json.loads(json.dumps(to_jsonable(_BARE_PLAN)))
    node = recorded
    for key in path[:-1]:
        node = node[key]
    return recorded, path[-1], node


class TestPlanFingerprint:
    def test_identical_declarations_share_a_fingerprint(self) -> None:
        """Two separately built but equal plans hash the same, so a snapshot can be matched later."""
        first = AnalysisPlan("return", (BakeoffAnalysis("h", ("a", "b")),), ExperimentDimensions())
        second = AnalysisPlan("return", (BakeoffAnalysis("h", ("a", "b")),), ExperimentDimensions())

        assert plan_fingerprint(first) == plan_fingerprint(second)

    def test_any_change_to_the_plan_changes_its_fingerprint(self) -> None:
        """Changing the measure after the fact must be detectable."""
        assert plan_fingerprint(_BARE_PLAN) != plan_fingerprint(replace(_BARE_PLAN, measure=TailMean(0.1)))

    def test_a_field_spelled_out_at_its_default_is_not_hashed(self) -> None:
        """Only declarations are hashed: writing a default out explicitly is the same plan as leaving it implicit."""
        explicit = AnalysisPlan(
            "return",
            (BakeoffAnalysis("h", ("a", "b"), min_seeds=5, measure=None),),
            ExperimentDimensions(condition="condition_name"),
            stats=BakeoffSpec(alpha=0.05, random_seed=0),
            higher_is_better=True,
        )

        assert plan_fingerprint(explicit) == plan_fingerprint(_BARE_PLAN)

    def test_a_declared_value_that_differs_from_its_default_is_hashed(self) -> None:
        """A changed statistic nested inside the plan is a changed plan."""
        tightened = replace(_BARE_PLAN, stats=BakeoffSpec(alpha=0.01))

        assert plan_fingerprint(tightened) != plan_fingerprint(_BARE_PLAN)

    def test_every_measure_declaration_hashes_distinctly(self) -> None:
        """Kind tags keep union members apart, and an analysis override differs from inheriting the plan's measure."""
        plans = [
            _BARE_PLAN,
            replace(_BARE_PLAN, measure=ReturnRate()),
            replace(_BARE_PLAN, measure=TailMean(0.1)),
            replace(_BARE_PLAN, measure=TailMean(0.2)),
            replace(_BARE_PLAN, measure=StoredValue()),
            replace(_BARE_PLAN, analyses=(BakeoffAnalysis("h", ("a", "b"), measure=FinalValue()),)),
        ]

        assert len({plan_fingerprint(plan) for plan in plans}) == len(plans)

    def test_the_report_keeps_the_full_resolved_plan(self) -> None:
        """Defaults are left out of the fingerprint only; the report JSON still says what every setting was."""
        document = replace(exploratory_report(_dataset(_SCORES), Path("db"), []), plan=_BARE_PLAN)

        plan = json.loads(report_to_json(document))["plan"]

        assert plan["analyses"][0]["measure"] is None
        assert plan["stats"]["alpha"] == 0.05


class TestRecordedPlanFingerprint:
    def test_a_recorded_plan_matches_the_plan_it_records(self) -> None:
        """A snapshot stores the full resolved plan; reading it back yields the declared-only fingerprint."""
        plan = replace(_BARE_PLAN, stats=BakeoffSpec(alpha=0.01), measure=TailMean(0.1))

        assert recorded_plan_fingerprint(to_jsonable(plan)) == plan_fingerprint(plan)

    def test_stored_value_measure_round_trips_in_a_recorded_plan(self) -> None:
        """A scalar plan keeps its explicit measure through snapshot serialization.

        Its recorded fingerprint must match the declared plan.
        """
        plan = replace(_BARE_PLAN, measure=StoredValue())

        assert recorded_plan_fingerprint(to_jsonable(plan)) == plan_fingerprint(plan)

    @pytest.mark.parametrize(
        "path",
        [("non_finite_worst_case",), ("stats", "interval"), ("analyses", 0, "measure")],
        ids=["plan-field", "stats-field", "analysis-field"],
    )
    def test_a_field_added_with_a_default_keeps_old_fingerprints(self, path: tuple[str | int, ...]) -> None:
        """A snapshot recorded before a defaulted field existed reads as that default, not as a changed plan."""
        recorded, leaf, parent = _recorded_parent(path)
        del parent[leaf]

        assert recorded_plan_fingerprint(recorded) == plan_fingerprint(_BARE_PLAN)

    @pytest.mark.parametrize(
        ("path", "value", "problem"),
        [
            (("renamed_field",), 1, "AnalysisPlan has no field 'renamed_field'"),
            (("measure",), {"kind": "windowed"}, "'windowed'"),
            (("stats", "estimand"), "trimmed", "'trimmed'"),
            (("dimensions", "sweep_axes"), [{"value_type": "float"}], "SweepAxis is missing its required field 'name'"),
        ],
        ids=["unknown-field", "unknown-kind", "unknown-literal", "missing-required"],
    )
    def test_a_plan_the_schema_cannot_read_is_rejected(
        self, path: tuple[str, ...], value: JsonValue, problem: str
    ) -> None:
        """Never guess: a recorded plan that does not fit the current schema has no fingerprint."""
        recorded, leaf, parent = _recorded_parent(path)
        parent[leaf] = value

        with pytest.raises(PlanInterpretationError, match=re.escape(problem)):
            recorded_plan_fingerprint(recorded)


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

    def test_the_provenance_names_the_analysis_code(self) -> None:
        """The audit copy alone tells which research-analysis release and commit computed the numbers."""
        dataset = _dataset(_SCORES)

        text = _markdown(dataset, [])

        provenance = text[text.index("## Provenance") :]
        assert re.search(rf"- Analysis code: research-analysis {re.escape(version('research-analysis'))} at commit [0-9a-f]{{40}}", provenance)

    def test_the_interval_method_is_named_in_the_bakeoff_section(self) -> None:
        """A reader of the markdown alone can tell an expanded percentile CI from a plain percentile one."""
        dataset = _dataset(_SCORES)
        report = bakeoff_report(
            dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50, interval="expanded_percentile")
        )

        text = _markdown(dataset, [BakeoffResult("headline", report)])

        assert "- Expanded percentile bootstrap, 50 resamples, seed 0" in text
        assert "paired expanded percentile bootstrap" in text

    def test_a_shared_referent_is_shown_as_reused_across_its_axis(self) -> None:
        """The markdown says which runs were reused in several blocks, and why the omnibus test is absent."""
        runs = pl.DataFrame(
            {
                "condition_name": ["shared"] * 4 + ["adversarial"] * 8,
                "seed": [*range(4), *range(4), *range(4)],
                "run_id": list(range(12)),
                "execution_id": list(range(12)),
                "ablation": pl.Series([None] * 12, dtype=pl.String),
                "eta": pl.Series([None] * 4 + [2.0] * 4 + [5.0] * 4, dtype=pl.Float64),
            }
        )
        metrics = runs.select("condition_name", "seed", "run_id", "execution_id").with_columns(
            step=pl.lit(0), metric=pl.lit("return"), value=pl.Series([float(index % 5) for index in range(12)])
        )
        eta = SweepAxis("eta", SweepValueType.FLOAT, when={"condition_name": ("adversarial",)})
        dataset = ExperimentDataset("demo", ExperimentDimensions(sweep_axes=(eta,)), runs, metrics, ())
        report = bakeoff_report(
            dataset, "return", ["adversarial", "shared"], block_by=["eta"], shared_across=["eta"], spec=BakeoffSpec(n_resamples=50)
        )

        text = _markdown(dataset, [BakeoffResult("headline", report)])

        assert "Shared across `eta`: shared lack `eta`, so their runs are reused in every `eta` block; 4 distinct runs appear in the reported cells (8 cell entries)" in text
        assert "No omnibus test: shared referent runs appear in several blocks" in text

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

    def test_a_champion_difference_warning_is_rendered_and_serialized(self) -> None:
        """The champion-difference caveat reaches both the markdown reader and the JSON consumer."""
        warning = small_sample_coverage_warning(5, "champion difference")
        assert warning is not None
        difference = HyperparameterDifferenceReport(
            minuend_config={"lr": 0.1},
            subtrahend_config={"lr": 0.2},
            mean_diff=1.0,
            estimate_ci=(0.5, 1.5),
            warnings=(warning,),
        )
        results: list[AnalysisResult] = [HypersDifferenceResult("gap", difference)]

        text = _markdown(_dataset(_SCORES), results)
        payload = json.loads(report_to_json(exploratory_report(_dataset(_SCORES), Path("db"), results)))

        assert "`small_sample_coverage` (all) : the champion difference CI rests on 5 runs" in text
        assert [w["code"] for w in payload["analyses"][0]["report"]["warnings"]] == ["small_sample_coverage"]


    def test_the_declared_plan_is_rendered_so_the_markdown_is_self_auditable(self) -> None:
        """A reader of report.md alone can see which metric, statistics, and analyses produced the numbers."""
        plan = AnalysisPlan(
            "return",
            (
                BakeoffAnalysis("headline", ("base", "new"), pairs=against_baseline(("base", "new"), "base")),
                BestHypersAnalysis("tuned", ("algorithm",)),
                HypersDifferenceAnalysis("gap", "algorithm", "ppo", "dqn"),
                SliceHypersAnalysis("sensitivity", "lr"),
                BakeoffAnalysis("whole_run", ("base", "new"), measure=ReturnRate()),
                BakeoffAnalysis("stored_run", ("base", "new"), measure=StoredValue()),
            ),
            ExperimentDimensions(sweep_axes=(SweepAxis("lr", SweepValueType.FLOAT),)),
            measure=TailMean(0.1),
            stats=BakeoffSpec(estimand="median", n_resamples=500, missingness="blockwise", random_seed=7),
            higher_is_better=False,
            non_finite_worst_case=-100.0,
        )
        document = replace(exploratory_report(_dataset(_SCORES), Path("db"), []), plan=plan)

        text = render_markdown(document)

        plan_text = text[text.index("## Declared plan") : text.index("## Provenance")]
        for expected in (
            "`return` (lower is better), per run: mean over the final 10% of logged steps; estimand: median",
            "condition `condition_name`, environment `none`, swept: lr (float)",
            "95% percentile bootstrap, 500 resamples, seed 7, min samples 3; missingness: blockwise",
            "Diverged runs: score -100",
            "`headline`: bakeoff of base, new; pairs: new − base; blocked by: nothing; shared across: nothing; "
            "champion over: none; untuned: none",
            "`tuned`: champion per algorithm",
            "`gap`: champion difference algorithm=ppo minus algorithm=dqn",
            "`sensitivity`: sensitivity to lr per whole grid",
            "`whole_run`: bakeoff of base, new; pairs: every pair; blocked by: nothing; shared across: nothing; "
            "champion over: none; untuned: none; min seeds 5; per run: return rate (mean over every logged point)",
            "`stored_run`: bakeoff of base, new; pairs: every pair; blocked by: nothing; shared across: nothing; "
            "champion over: none; untuned: none; min seeds 5; per run: stored per-run scalar",
        ):
            assert expected in plan_text, expected

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


class TestReportFiles:
    def test_a_written_report_holds_its_json_markdown_and_figures(self, tmp_path: Path) -> None:
        """The JSON and markdown on disk describe the same document, including the figures beside them."""
        dataset = _dataset(_SCORES)
        report = bakeoff_report(dataset, "return", ["base", "new"], spec=BakeoffSpec(n_resamples=50))
        document = exploratory_report(dataset, Path("db"), [BakeoffResult("headline", report)])
        output_dir = tmp_path / "reports" / "demo" / "now"

        written = write_report_document(document, output_dir)

        assert json.loads((output_dir / "report.json").read_text()) == json.loads(report_to_json(written))
        assert (output_dir / "report.md").read_text() == render_markdown(written)
        assert all((output_dir / path).exists() for path in written.plots.values())

    def test_the_default_directory_is_timestamped_under_the_results_root(self, tmp_path: Path) -> None:
        """Each report gets its own directory under reports/<experiment>, with unsafe characters replaced."""
        directory = report_directory(tmp_path, "exp 003/lr")

        assert directory.parent == tmp_path / "reports" / "exp_003_lr"
        assert re.fullmatch(r"\d{8}T\d{6}Z", directory.name)
