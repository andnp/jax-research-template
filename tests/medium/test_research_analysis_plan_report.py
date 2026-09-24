"""Integration tests: a declared plan run against a real experiment database becomes a trustworthy report."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from experiment_definition.db import DatabaseManager
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import ExperimentDimensions
from research_analysis.diagnostics.bakeoff import IncompleteCell
from research_analysis.plan import AnalysisPlan, BakeoffAnalysis
from research_analysis.plan_report import run_plan
from research_analysis.plan_snapshots import PLAN_HISTORY_FILE, PlanSnapshot, read_plan_snapshots, record_plan_snapshot
from research_analysis.report import PlanStatus, ReportStatus
from research_analysis.report_json import plan_fingerprint

STARTED = "2026-01-02T00:00:00+00:00"
BEFORE = datetime(2026, 1, 1, tzinfo=UTC)
AFTER = datetime(2026, 1, 3, tzinfo=UTC)

PLAN = AnalysisPlan(
    "return",
    (BakeoffAnalysis("headline", ("base", "new")),),
    ExperimentDimensions(),
    stats=BakeoffSpec(n_resamples=100),
)


class _Experiment:
    def __init__(self, root: Path) -> None:
        self.db_path = root / "experiments.sqlite"
        self.metrics_path = root / "metrics.sqlite"
        self.root = root
        with DatabaseManager(self.db_path) as database:
            database.initialize()
            self.experiment_id = database.add_experiment("demo")
            self.algo = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-hash")
            self.env = database.add_component_version(database.add_component("cartpole", "ENV"), "env-hash")
        with sqlite3.connect(self.metrics_path) as connection:
            connection.execute(
                "CREATE TABLE metrics (run_id INTEGER, execution_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
            )

    def add_run(self, condition: str, seed: int, value: float | None, *, status: str = "COMPLETED", artifacts: bool = True) -> int:
        with DatabaseManager(self.db_path) as database:
            hyper_id = database.add_hyperparam_config({"condition_name": condition, "seed": seed})
            run_id = database.add_run(self.experiment_id, self.algo, self.env, hyper_id, seed)
            execution_id = database.add_execution(git_commit="abc123")
            database.link_execution_run(execution_id, run_id)
            if artifacts:
                database.record_execution_artifacts(
                    execution_id, str(self.root / str(execution_id)), metadata={"metrics_db_path": str(self.metrics_path)}
                )
            database.update_execution_status(execution_id, status, start_time=STARTED)
        if status == "COMPLETED" and artifacts:
            with sqlite3.connect(self.metrics_path) as connection:
                connection.execute("INSERT INTO metrics VALUES (?, ?, 'return', 0, ?)", (run_id, execution_id, value))
        return run_id


def _complete(root: Path, seeds: int = 5) -> _Experiment:
    experiment = _Experiment(root)
    for seed in range(seeds):
        experiment.add_run("base", seed, float(seed))
        experiment.add_run("new", seed, float(seed) + 1.0 + 0.1 * seed)
    return experiment


def _declared_before() -> tuple[PlanSnapshot, ...]:
    return (PlanSnapshot(plan_fingerprint(PLAN), BEFORE, "{}"),)


class TestPlanStatus:
    def test_a_plan_recorded_before_the_runs_yields_a_declared_report(self, tmp_path: Path) -> None:
        """Running the recorded plan verbatim is the one path to a declared label."""
        experiment = _complete(tmp_path)

        document = run_plan(experiment.db_path, "demo", PLAN, _declared_before())

        assert (document.status, document.plan_status, document.exploratory_reasons) == (
            ReportStatus.OK,
            PlanStatus.DECLARED,
            (),
        )
        assert [result.name for result in document.analyses] == ["headline"]
        assert {item.git_commit for item in document.provenance.executions} == {"abc123"}

    def test_a_plan_never_recorded_is_exploratory(self, tmp_path: Path) -> None:
        """Without a snapshot the plan may have been written after the results were seen."""
        experiment = _complete(tmp_path)

        document = run_plan(experiment.db_path, "demo", PLAN, ())

        assert document.plan_status is PlanStatus.EXPLORATORY
        assert document.exploratory_reasons[0].startswith("no plan snapshot before execution")

    def test_a_snapshot_taken_only_after_the_runs_is_exploratory(self, tmp_path: Path) -> None:
        """Recording the plan once results exist does not make it pre-registered."""
        experiment = _complete(tmp_path)

        document = run_plan(experiment.db_path, "demo", PLAN, (PlanSnapshot(plan_fingerprint(PLAN), AFTER, "{}"),))

        assert document.exploratory_reasons[0].startswith("no plan snapshot before execution")

    def test_a_plan_edited_after_the_runs_began_is_exploratory(self, tmp_path: Path) -> None:
        """Switching the measure after launch is metric shopping unless it is labelled as such."""
        experiment = _complete(tmp_path)
        original = replace(PLAN, stats=BakeoffSpec(n_resamples=100, alpha=0.1))
        history = (PlanSnapshot(plan_fingerprint(original), BEFORE, "{}"), PlanSnapshot(plan_fingerprint(PLAN), AFTER, "{}"))

        document = run_plan(experiment.db_path, "demo", PLAN, history)

        assert document.plan_status is PlanStatus.EXPLORATORY
        assert document.exploratory_reasons[0].startswith("plan changed after results were recorded")
        assert document.status is ReportStatus.OK


class TestFailedReports:
    def test_a_fatal_load_reports_diagnostics_and_no_results(self, tmp_path: Path) -> None:
        """A completed run with no artifact record is fatal: the document explains it and holds no numbers."""
        experiment = _complete(tmp_path)
        experiment.add_run("new", 9, None, artifacts=False)

        document = run_plan(experiment.db_path, "demo", PLAN, _declared_before())

        assert document.status is ReportStatus.LOAD_FAILED
        assert document.analyses == ()
        assert [item.code for item in document.fatal_diagnostics] == ["malformed_artifact"]
        assert sum(cell.planned for cell in document.accounting.cells) == 11

    def test_a_diverged_seed_fails_the_analysis_naming_the_run(self, tmp_path: Path) -> None:
        """A NaN score is an outcome, so without a declared worst case the report fails and names the run."""
        experiment = _complete(tmp_path)
        diverged = experiment.add_run("new", 5, math.nan)
        experiment.add_run("base", 5, 5.0)

        document = run_plan(experiment.db_path, "demo", PLAN, _declared_before())

        assert document.status is ReportStatus.ANALYSIS_FAILED
        assert document.analyses == ()
        assert document.failure is not None
        assert [run["run_id"] for run in document.failure.non_finite_runs] == [diverged]
        assert {cell.diverged.get("return", 0) for cell in document.accounting.cells} == {0, 1}

    def test_a_declared_worst_case_scores_the_diverged_seed(self, tmp_path: Path) -> None:
        """With a declared worst case the diverged run counts at that value and stays in the accounting."""
        experiment = _complete(tmp_path)
        experiment.add_run("new", 5, math.nan)
        experiment.add_run("base", 5, 5.0)
        plan = replace(PLAN, non_finite_worst_case=-100.0)
        snapshots = (PlanSnapshot(plan_fingerprint(plan), BEFORE, "{}"),)

        document = run_plan(experiment.db_path, "demo", plan, snapshots)

        assert document.status is ReportStatus.OK
        assert document.plan_status is PlanStatus.DECLARED
        assert any(cell.diverged.get("return") == 1 for cell in document.accounting.cells)

    def test_a_crashed_seed_fails_the_analysis_naming_the_cell(self, tmp_path: Path) -> None:
        """A failed seed under missingness='error' produces the incomplete cell and its failed count."""
        experiment = _complete(tmp_path)
        experiment.add_run("base", 5, 5.0)
        experiment.add_run("new", 5, None, status="FAILED")

        document = run_plan(experiment.db_path, "demo", PLAN, _declared_before())

        assert document.status is ReportStatus.ANALYSIS_FAILED
        assert document.failure is not None
        assert document.failure.incomplete_cells == (IncompleteCell("new", "", (5,), failed=1, pending=0, excluded=0),)


class TestPlanHistory:
    def test_history_appends_each_changed_plan_and_never_rewrites(self, tmp_path: Path) -> None:
        """Re-recording the current plan is a no-op; a changed plan is appended after the earlier entries."""
        record_plan_snapshot(PLAN, tmp_path)
        first_line = (tmp_path / PLAN_HISTORY_FILE).read_text()
        record_plan_snapshot(PLAN, tmp_path)
        changed = replace(PLAN, metric="loss")
        record_plan_snapshot(changed, tmp_path)

        history = read_plan_snapshots(tmp_path)

        assert [item.fingerprint for item in history] == [plan_fingerprint(PLAN), plan_fingerprint(changed)]
        assert (tmp_path / PLAN_HISTORY_FILE).read_text().startswith(first_line)
        assert json.loads(history[1].plan_json)["metric"] == "loss"

    def test_no_history_reads_as_empty(self, tmp_path: Path) -> None:
        """A results root that never recorded a plan has an empty history, not an error."""
        assert read_plan_snapshots(tmp_path) == ()

