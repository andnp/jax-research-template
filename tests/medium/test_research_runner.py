"""Medium integration tests for research-runner — plan-execute-record lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiment_definition.component import Component, ComponentType
from experiment_definition.db import DatabaseManager, PlannedExecution
from experiment_definition.experiment import Experiment
from experiment_definition.status import RunState, run_state
from research_runner.git import capture_git_metadata
from research_runner.runner import execute_batch, run_experiment
from research_runner.types import ExecutionContext, ExecutionResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _unsatisfied_run_ids(database: DatabaseManager, experiment_id: int):
    statuses = database.list_run_execution_statuses(experiment_id)
    return {run_id for run_id, run_statuses in statuses.items() if run_state(run_statuses) is not RunState.COMPLETED}


def _trivial_train_fn(ctx: ExecutionContext) -> ExecutionResult:
    return ExecutionResult()


def _make_experiment(name: str = "TestSweep") -> Experiment:
    algo = Component(name="TestAlgo", path=Path("/nonexistent/algo.py"), type=ComponentType.ALGO)
    env = Component(name="TestEnv", path=Path("/nonexistent/env.py"), type=ComponentType.ENV)
    exp = Experiment(name, description="test experiment")
    exp.add_parameter("seed", [0, 1])
    with exp.for_component(algo):
        exp.add_parameter("lr", [1e-3])
    with exp.for_component(env):
        exp.add_parameter("gamma", [0.99])
    return exp


# ── run_experiment ────────────────────────────────────────────────────────────


class TestRunExperiment:
    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "experiments.sqlite"

    @pytest.fixture()
    def executions_root(self, tmp_path: Path) -> Path:
        return tmp_path / "results" / "executions"

    def test_happy_path(self, db_path: Path, executions_root: Path) -> None:
        """
        Full lifecycle: sync experiment, plan, execute with trivial callback,
        verify execution roots, DB status, manifest, and artifacts.
        """
        exp = _make_experiment()
        roots = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )

        assert isinstance(roots, list)
        assert len(roots) >= 1
        assert all(isinstance(r, Path) for r in roots)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("TestSweep")
            assert experiment_row is not None

            for root in roots:
                manifest_path = root / "manifest.json"
                assert manifest_path.exists(), f"Manifest missing at {manifest_path}"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                assert manifest["experiment_slug"] == "TestSweep"

                exec_id = manifest["execution_id"]
                execution = database.get_execution(exec_id)
                assert execution is not None
                assert execution.status == "COMPLETED"

                artifacts = database.get_execution_artifacts(exec_id)
                assert artifacts is not None
                assert artifacts.root_path == str(root)
                artifact_metadata = json.loads(artifacts.metadata_json or "{}")
                assert artifact_metadata["run_ids"] == manifest["run_ids"]
                assert "static_config_json" in artifact_metadata
                assert "vmap_zone_json" in artifact_metadata
                assert artifact_metadata["metrics_db_path"] == manifest["metrics_db_path"]

    def test_on_batch_complete_invoked(self, db_path: Path, executions_root: Path) -> None:
        """
        The on_batch_complete callback receives each execution root as it finishes.
        """
        exp = _make_experiment()
        completed_roots: list[Path] = []

        roots = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
            on_batch_complete=lambda root: completed_roots.append(root),
        )

        assert len(completed_roots) == len(roots)
        assert completed_roots == roots

    def test_already_satisfied_returns_empty(self, db_path: Path, executions_root: Path) -> None:
        """
        Running the same experiment twice returns an empty list because all
        runs are already COMPLETED.
        """
        exp = _make_experiment()
        first = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )
        assert len(first) >= 1

        second = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )
        assert second == []


    def test_only_restricts_which_runs_execute(self, db_path: Path, executions_root: Path) -> None:
        """A selector confines the loop to matching runs, leaving the rest plannable.

        The unselected runs must stay executable afterwards: ``--only`` narrows one
        invocation, it does not retire what it skipped.
        """
        exp = _make_experiment()

        selected = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
            only={"seed": "0"},
        )
        assert len(selected) == 1

        with DatabaseManager(db_path) as database:
            database.initialize()
            runs = database.list_execution_runs(
                database.list_executions()[0].id,
            )
        assert [run.seed for run in runs] == [0]

        remaining = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )
        assert len(remaining) == 1

    def test_only_matching_nothing_executes_nothing(self, db_path: Path, executions_root: Path) -> None:
        """An unmatched selector is a no-op rather than a fallback to running everything."""
        exp = _make_experiment()

        roots = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
            only={"seed": "99"},
        )

        assert roots == []


# ── execute_batch ─────────────────────────────────────────────────────────────


class TestExecuteBatch:
    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "experiments.sqlite"

    @pytest.fixture()
    def executions_root(self, tmp_path: Path) -> Path:
        return tmp_path / "results" / "executions"

    def _plan_one_batch(self, db_path: Path, executions_root: Path) -> PlannedExecution:
        exp = _make_experiment()
        exp.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("TestSweep")
            assert experiment_row is not None
            planned = database.plan_experiment_execution_batches(
                experiment_row.id,
                executions_root,
            )
            assert len(planned) >= 1
        return planned[0]

    def test_happy_path(self, db_path: Path, executions_root: Path) -> None:
        """
        Manually plan a batch then execute it; verify status transitions
        and manifest creation.
        """
        planned = self._plan_one_batch(db_path, executions_root)

        root = execute_batch(
            db_path,
            planned,
            _trivial_train_fn,
            capture_git=False,
        )

        assert root == Path(planned.root_path)
        assert (root / "manifest.json").exists()

        with DatabaseManager(db_path) as database:
            database.initialize()
            execution = database.get_execution(planned.execution_id)
            assert execution is not None
            assert execution.status == "COMPLETED"

    def test_failure_sets_failed_status(self, db_path: Path, executions_root: Path) -> None:
        """
        When the train callback raises, the execution status is set to FAILED
        and the exception propagates.
        """
        planned = self._plan_one_batch(db_path, executions_root)

        def _failing_train_fn(ctx: ExecutionContext) -> ExecutionResult:
            raise RuntimeError("Training exploded")

        with pytest.raises(RuntimeError, match="Training exploded"):
            execute_batch(
                db_path,
                planned,
                _failing_train_fn,
                capture_git=False,
            )

        with DatabaseManager(db_path) as database:
            database.initialize()
            execution = database.get_execution(planned.execution_id)
            assert execution is not None
            assert execution.status == "FAILED"

    def test_static_config_mismatch_sets_failed_status(
        self, db_path: Path, executions_root: Path,
    ) -> None:
        """A preflight static-config error fails the claimed execution."""
        planned = self._plan_one_batch(db_path, executions_root)
        planned = planned._replace(static_config={"missing": "value"})

        with pytest.raises(ValueError, match="static parameter"):
            execute_batch(
                db_path,
                planned,
                _trivial_train_fn,
                capture_git=False,
            )

        with DatabaseManager(db_path) as database:
            database.initialize()
            execution = database.get_execution(planned.execution_id)
            assert execution is not None
            assert execution.status == "FAILED"

    def test_failure_preserves_original_error_after_failed_transition(
        self, db_path: Path, executions_root: Path,
    ) -> None:
        """A failed status transition does not replace the completion error."""
        planned = self._plan_one_batch(db_path, executions_root)

        def _failing_train_fn(ctx: ExecutionContext) -> ExecutionResult:
            with DatabaseManager(db_path) as database:
                database.initialize()
                database.update_execution_status(ctx.execution_id, "FAILED")
            return ExecutionResult()

        with pytest.raises(ValueError, match="FAILED -> COMPLETED"):
            execute_batch(
                db_path,
                planned,
                _failing_train_fn,
                capture_git=False,
            )





class TestCaptureGitMetadata:
    def test_returns_tuple_of_two(self) -> None:
        """
        capture_git_metadata returns a 2-tuple; each element is str or None.
        """
        commit, diff = capture_git_metadata()
        assert isinstance(commit, (str, type(None)))
        assert isinstance(diff, (str, type(None)))


# ── mid-sweep crash recovery ─────────────────────────────────────────────────


class TestMidSweepCrashRecovery:
    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "experiments.sqlite"

    @pytest.fixture()
    def executions_root(self, tmp_path: Path) -> Path:
        return tmp_path / "results" / "executions"

    def _make_two_batch_experiment(self) -> Experiment:
        algo = Component(name="TestAlgo", path=Path("/nonexistent/algo.py"), type=ComponentType.ALGO)
        env = Component(name="TestEnv", path=Path("/nonexistent/env.py"), type=ComponentType.ENV)
        exp = Experiment("CrashSweep", description="test experiment")
        exp.add_parameter("seed", [0])
        with exp.for_component(algo):
            exp.add_parameter("arch", ["cnn", "mlp"], is_static=True)
        with exp.for_component(env):
            exp.add_parameter("gamma", [0.99])
        return exp

    def test_resumes_after_first_batch_crashes(self, db_path: Path, executions_root: Path) -> None:
        """
        A sweep with 2 batches whose train_fn raises on its very first call
        must, after a retry, still reach every batch: the un-executed batch
        must never be stranded as an unplannable PENDING execution.
        """
        exp = self._make_two_batch_experiment()
        calls: list[str] = []

        def flaky_train_fn(ctx: ExecutionContext) -> ExecutionResult:
            calls.append(str(ctx.static_config["arch"]))
            if len(calls) == 1:
                raise RuntimeError("boom")
            return ExecutionResult()

        with pytest.raises(RuntimeError, match="boom"):
            run_experiment(
                db_path,
                exp,
                flaky_train_fn,
                executions_root=executions_root,
                capture_git=False,
            )

        roots = run_experiment(
            db_path,
            exp,
            flaky_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )

        assert set(calls) == {"cnn", "mlp"}
        assert len(roots) == 2

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("CrashSweep")
            assert experiment_row is not None
            assert _unsatisfied_run_ids(database, experiment_row.id) == set()


class TestDynamicArmCoverage:
    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "experiments.sqlite"

    @pytest.fixture()
    def executions_root(self, tmp_path: Path) -> Path:
        return tmp_path / "results" / "executions"

    def _make_static_plus_dynamic_experiment(self) -> Experiment:
        algo = Component(name="TestAlgo", path=Path("/nonexistent/algo.py"), type=ComponentType.ALGO)
        exp = Experiment("ArmSweep", description="test experiment")
        exp.add_parameter("seed", [0, 1])
        with exp.for_component(algo):
            exp.add_parameter("arch", ["mlp"], is_static=True)
            exp.add_parameter("lr", [1e-3, 3e-4])
        return exp

    def test_every_swept_arm_reaches_train_fn(self, db_path: Path, executions_root: Path) -> None:
        """
        A batch keyed on the static config may still hold several dynamic arms.
        Every declared (lr, seed) pair must be handed to train_fn, and no run
        may be recorded as satisfied without having been trained.
        """
        exp = self._make_static_plus_dynamic_experiment()
        seen: set[tuple[float, int]] = set()
        stacked_lrs: set[float] = set()
        axis_keys: list[list[str]] = []

        def recording_train_fn(ctx: ExecutionContext) -> ExecutionResult:
            assert ctx.static_config["arch"] == "mlp"
            axes = ctx.axes()
            axis_keys.append(sorted(axes))
            for stacked in axes.get("lr", []):
                assert isinstance(stacked, float)
                stacked_lrs.add(stacked)
            for point in ctx.points:
                lr = point.hyperparameters["lr"]
                seed = point.hyperparameters["seed"]
                assert isinstance(lr, float)
                assert isinstance(seed, int)
                seen.add((lr, seed))
            return ExecutionResult()

        run_experiment(
            db_path,
            exp,
            recording_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )

        assert seen == {(1e-3, 0), (1e-3, 1), (3e-4, 0), (3e-4, 1)}

        assert stacked_lrs == {1e-3, 3e-4}
        assert all("arch" not in keys for keys in axis_keys)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("ArmSweep")
            assert experiment_row is not None
            assert _unsatisfied_run_ids(database, experiment_row.id) == set()


# ── stale (orphaned) execution reclaim ───────────────────────────────────────


class TestStaleExecutionReclaim:
    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "experiments.sqlite"

    @pytest.fixture()
    def executions_root(self, tmp_path: Path) -> Path:
        return tmp_path / "results" / "executions"

    def _make_single_run_experiment(self) -> Experiment:
        algo = Component(name="TestAlgo", path=Path("/nonexistent/algo.py"), type=ComponentType.ALGO)
        env = Component(name="TestEnv", path=Path("/nonexistent/env.py"), type=ComponentType.ENV)
        exp = Experiment("StaleSweep", description="test experiment")
        exp.add_parameter("seed", [0])
        with exp.for_component(algo):
            exp.add_parameter("lr", [1e-3])
        with exp.for_component(env):
            exp.add_parameter("gamma", [0.99])
        return exp

    def _orphan_running_execution(self, db_path: Path, executions_root: Path) -> None:
        """
        Plan one batch and leave it RUNNING with an ancient start_time,
        simulating a process that was SIGKILLed / OOM-killed / lost its
        working directory before it could reach execute_batch's except
        handler and mark the execution FAILED.
        """
        exp = self._make_single_run_experiment()
        exp.sync(db_path)
        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("StaleSweep")
            assert experiment_row is not None
            planned = database.plan_next_execution_batch(experiment_row.id, executions_root)
            assert planned is not None
            database.update_execution_status(
                planned.execution_id, "RUNNING", start_time="2000-01-01T00:00:00+00:00",
            )

    def test_hard_killed_execution_stalls_sweep_without_reclaim(
        self, db_path: Path, executions_root: Path,
    ) -> None:
        """
        Without reclaim_stale_after_seconds, a run held by a stuck RUNNING
        execution is neither satisfied nor replannable: the sweep silently
        stalls short of completion instead of resuming.
        """
        self._orphan_running_execution(db_path, executions_root)
        exp = self._make_single_run_experiment()

        roots = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
        )

        assert roots == []
        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("StaleSweep")
            assert experiment_row is not None
            assert _unsatisfied_run_ids(database, experiment_row.id) != set()

    def test_reclaim_stale_after_seconds_resumes_orphaned_run(
        self, db_path: Path, executions_root: Path,
    ) -> None:
        """
        Passing reclaim_stale_after_seconds reclaims the orphaned RUNNING
        execution (its start_time is far older than the threshold) so the
        run it was holding actually gets replanned and completed.
        """
        self._orphan_running_execution(db_path, executions_root)
        exp = self._make_single_run_experiment()

        roots = run_experiment(
            db_path,
            exp,
            _trivial_train_fn,
            executions_root=executions_root,
            capture_git=False,
            reclaim_stale_after_seconds=60,
        )

        assert len(roots) == 1
        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("StaleSweep")
            assert experiment_row is not None
            assert _unsatisfied_run_ids(database, experiment_row.id) == set()
