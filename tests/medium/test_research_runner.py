"""Medium integration tests for research-runner — plan-execute-record lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiment_definition.component import Component, ComponentType
from experiment_definition.db import DatabaseManager, PlannedExecution
from experiment_definition.experiment import Experiment
from research_runner.git import capture_git_metadata
from research_runner.runner import _derive_metrics_db_path, execute_batch, run_experiment
from research_runner.types import ExecutionContext, ExecutionResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _trivial_train_fn(ctx: ExecutionContext) -> ExecutionResult:
    return ExecutionResult(metadata={"trained": True})


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
                assert manifest["metadata"] == {"trained": True}

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
                assert artifact_metadata["trained"] is True
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


# ── _derive_metrics_db_path ──────────────────────────────────────────────────


class TestDeriveMetricsDbPath:
    def test_with_executions_ancestor(self) -> None:
        """
        When execution_root contains an 'executions' ancestor directory,
        metrics.sqlite is placed next to it.
        """
        root = Path("/tmp/results/executions/123")
        result = _derive_metrics_db_path(root)
        assert result == Path("/tmp/results/metrics.sqlite")

    def test_without_executions_ancestor_raises(self) -> None:
        """
        When execution_root has no 'executions' ancestor, raises ValueError
        to force explicit metrics_db_path.
        """
        root = Path("/tmp/some/other/path")
        with pytest.raises(ValueError, match="Pass metrics_db_path explicitly"):
            _derive_metrics_db_path(root)

    def test_executions_is_root_itself(self) -> None:
        """
        When the execution_root directory is itself named 'executions',
        metrics.sqlite is placed in its parent.
        """
        root = Path("/tmp/results/executions")
        result = _derive_metrics_db_path(root)
        assert result == Path("/tmp/results/metrics.sqlite")


# ── capture_git_metadata ─────────────────────────────────────────────────────


class TestCaptureGitMetadata:
    def test_returns_tuple_of_two(self) -> None:
        """
        capture_git_metadata returns a 2-tuple; each element is str or None.
        """
        commit, diff = capture_git_metadata()
        assert isinstance(commit, (str, type(None)))
        assert isinstance(diff, (str, type(None)))
