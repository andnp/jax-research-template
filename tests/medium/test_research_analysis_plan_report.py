"""Integration tests: a declared plan run against a real experiment database becomes a trustworthy report."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from experiment_definition.db import DatabaseManager
from experiment_definition.parameter import ParameterValue
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.types import ExperimentDimensions, SweepAxis, SweepValueType
from research_analysis.diagnostics.bakeoff import IncompleteCell
from research_analysis.plan import AnalysisPlan, BakeoffAnalysis, against_baseline
from research_analysis.plan_report import run_plan
from research_analysis.plan_snapshots import PLAN_HISTORY_FILE, PlanSnapshot, read_plan_snapshots, record_plan_snapshot
from research_analysis.report import BakeoffResult, PlanStatus, ReportStatus
from research_analysis.report_json import plan_fingerprint, report_to_json

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

    def add_run(
        self,
        condition: str,
        seed: int,
        value: float | None,
        *,
        status: str = "COMPLETED",
        artifacts: bool = True,
        started: str = STARTED,
        hyperparameters: dict[str, ParameterValue] | None = None,
    ) -> int:
        with DatabaseManager(self.db_path) as database:
            hyper_id = database.add_hyperparam_config({**(hyperparameters or {}), "condition_name": condition, "seed": seed})
            run_id = database.add_run(self.experiment_id, self.algo, self.env, hyper_id, seed)
            execution_id = database.add_execution(git_commit="abc123")
            database.link_execution_run(execution_id, run_id)
            if artifacts:
                database.record_execution_artifacts(
                    execution_id, str(self.root / str(execution_id)), metadata={"metrics_db_path": str(self.metrics_path)}
                )
            database.update_execution_status(execution_id, status, start_time=started)
        if status == "COMPLETED" and artifacts:
            with sqlite3.connect(self.metrics_path) as connection:
                connection.execute("INSERT INTO metrics VALUES (?, ?, 'return', 0, ?)", (run_id, execution_id, value))
        return run_id


def _complete(root: Path, seeds: int = 5, started: str = STARTED) -> _Experiment:
    experiment = _Experiment(root)
    for seed in range(seeds):
        experiment.add_run("base", seed, float(seed), started=started)
        experiment.add_run("new", seed, float(seed) + 1.0 + 0.1 * seed, started=started)
    return experiment


def _declared_before() -> tuple[PlanSnapshot, ...]:
    return (PlanSnapshot(plan_fingerprint(PLAN), BEFORE, "{}"),)


ARMS = ("adversarial", "ortho", "shared")
PER_ARM_PLAN = AnalysisPlan(
    "return",
    (
        BakeoffAnalysis(
            "per_eta", ARMS, pairs=against_baseline(ARMS, "shared"), block_by=("eta",), shared_across=("eta",)
        ),
        BakeoffAnalysis(
            "tuned", ARMS, pairs=against_baseline(ARMS, "shared"), select_champion_over=("eta",), untuned=("shared",)
        ),
    ),
    ExperimentDimensions(
        sweep_axes=(SweepAxis("eta", SweepValueType.FLOAT, when={"condition_name": ("adversarial", "ortho")}),)
    ),
    stats=BakeoffSpec(n_resamples=100),
)


class TestPerArmAxisPlan:
    def test_an_axis_on_several_arms_with_a_fixed_referent_is_one_declared_plan(self, tmp_path: Path) -> None:
        """One dataset, one plan: the referent joins every eta block, then stands untuned beside eta-tuned arms."""
        experiment = _Experiment(tmp_path)
        for seed in range(5):
            experiment.add_run("shared", seed, float(seed))
            for condition, gain in (("adversarial", 1.0), ("ortho", 2.0)):
                for eta in (2.0, 5.0):
                    experiment.add_run(condition, seed, seed + gain * eta, hyperparameters={"eta": eta})
        snapshots = (PlanSnapshot(plan_fingerprint(PER_ARM_PLAN), BEFORE, "{}"),)

        document = run_plan(experiment.db_path, "demo", PER_ARM_PLAN, snapshots)

        assert (document.status, document.plan_status) == (ReportStatus.OK, PlanStatus.DECLARED)
        per_eta, tuned = (result.report for result in document.analyses if isinstance(result, BakeoffResult))
        assert per_eta.shared_across == {"eta": ("shared",)}
        assert per_eta.difference("ortho", "shared", "5.0").estimate == pytest.approx(10.0)
        assert tuned.untuned == ("shared",)
        assert tuned.difference("ortho", "shared").estimate == pytest.approx(10.0, abs=0.5)
        assert sum(cell.completed for cell in document.accounting.cells) == 25
        assert json.loads(report_to_json(document))["analyses"][0]["report"]["shared_across"] == {"eta": ["shared"]}


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

    def test_a_plan_reverted_after_an_edited_epoch_is_exploratory(self, tmp_path: Path) -> None:
        """Every execution is checked: runs launched under an edited plan stay exploratory after a revert."""
        experiment = _complete(tmp_path)
        edited = replace(PLAN, stats=BakeoffSpec(n_resamples=100, alpha=0.1))
        experiment.add_run("base", 5, 5.0, started="2026-01-04T00:00:00+00:00")
        experiment.add_run("new", 5, 6.5, started="2026-01-04T00:00:00+00:00")
        history = (PlanSnapshot(plan_fingerprint(PLAN), BEFORE, "{}"), PlanSnapshot(plan_fingerprint(edited), AFTER, "{}"))

        document = run_plan(experiment.db_path, "demo", PLAN, history)

        late = [item.execution_id for item in document.provenance.executions if item.start_time != STARTED]
        assert document.plan_status is PlanStatus.EXPLORATORY
        assert document.exploratory_reasons == (
            f"plan changed after results were recorded: the plan in force at these executions was "
            f"{plan_fingerprint(edited)} (executions {', '.join(map(str, sorted(late)))})",
        )


class TestExecutionStartTimes:
    @pytest.mark.parametrize(
        "started",
        ["2026-01-02T01:00:00Z", "2026-01-01T20:00:00-05:00"],
        ids=["zulu", "non-utc-offset"],
    )
    def test_an_offset_qualified_start_is_ordered_in_utc(self, tmp_path: Path, started: str) -> None:
        """Both spellings name 01:00 UTC, after a 00:30 UTC snapshot, whatever the local clock read."""
        experiment = _complete(tmp_path, started=started)
        snapshot = PlanSnapshot(plan_fingerprint(PLAN), datetime(2026, 1, 2, 0, 30, tzinfo=UTC), "{}")

        document = run_plan(experiment.db_path, "demo", PLAN, (snapshot,))

        assert (document.plan_status, document.exploratory_reasons) == (PlanStatus.DECLARED, ())

    def test_a_naive_start_is_unknown_rather_than_guessed(self, tmp_path: Path) -> None:
        """A legacy start without an offset cannot be ordered against the plan history, so the report says so."""
        experiment = _complete(tmp_path, started="2026-01-02T00:00:00")

        document = run_plan(experiment.db_path, "demo", PLAN, _declared_before())

        ids = ", ".join(str(item.execution_id) for item in document.provenance.executions)
        assert document.plan_status is PlanStatus.EXPLORATORY
        assert document.exploratory_reasons == (
            "start time '2026-01-02T00:00:00' is not an ISO time with a UTC offset, "
            f"so its order against the plan history is unknown (executions {ids})",
        )

    def test_an_invalid_snapshot_is_exploratory_without_crashing(self, tmp_path: Path) -> None:
        """A malformed recorded_at from a hand-edited history file never crashes comparing it to execution starts."""
        experiment = _complete(tmp_path)
        invalid = PlanSnapshot(
            plan_fingerprint(PLAN), None, "{}", invalid_reason="recorded_at '2026-01-01T00:00:00' has no UTC offset"
        )

        document = run_plan(experiment.db_path, "demo", PLAN, (invalid,))

        assert document.plan_status is PlanStatus.EXPLORATORY
        assert document.exploratory_reasons[0] == "plan history has an unusable snapshot: recorded_at '2026-01-01T00:00:00' has no UTC offset"


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

    def test_too_few_seeds_fails_the_analysis_as_insufficient_data(self, tmp_path: Path) -> None:
        """Fewer seeds than min_samples is a data shortfall, not a crash: the report names what was needed."""
        experiment = _complete(tmp_path, seeds=1)

        document = run_plan(experiment.db_path, "demo", PLAN, _declared_before())

        assert document.status is ReportStatus.ANALYSIS_FAILED
        assert document.analyses == ()
        assert document.failure is not None
        assert document.failure.error_type == "InsufficientDataError"
        cell = document.failure.insufficient_data[0]
        assert (cell.required, cell.observed) == ("at least 3 blocks with at least two observed algorithms", "1")

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

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            ("2026-01-01T00:00:00", "recorded_at '2026-01-01T00:00:00' is not an ISO time with a UTC offset"),
            ("not-a-time", "recorded_at 'not-a-time' is not an ISO time"),
        ],
        ids=["naive", "malformed"],
    )
    def test_a_bad_recorded_at_is_read_as_invalid_not_guessed(self, tmp_path: Path, raw: str, reason: str) -> None:
        """A hand-edited history entry never raises: it comes back flagged instead of silently misordered."""
        entry = {"fingerprint": plan_fingerprint(PLAN), "recorded_at": raw, "plan": {}}
        (tmp_path / PLAN_HISTORY_FILE).write_text(json.dumps(entry) + "\n")

        history = read_plan_snapshots(tmp_path)

        assert history == (PlanSnapshot(plan_fingerprint(PLAN), None, "{}", invalid_reason=reason),)
