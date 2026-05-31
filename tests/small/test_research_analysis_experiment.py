"""Small tests for research_analysis.experiment — load_experiment_metrics and resolve_run_artifacts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from experiment_definition.db import DatabaseManager
from research_analysis.experiment import load_experiment_metrics, resolve_run_artifacts

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class _RunRecord:
    run_id: int
    execution_id: int
    seed: int
    condition_name: str


@dataclass
class _ExperimentFixture:
    runs: list[_RunRecord]
    metrics_db: Path


def _bootstrap_experiments_db(db_path: Path) -> _ExperimentFixture:
    """Create a minimal experiments DB with two conditions (oracle + mcar), one seed each."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_db = db_path.parent / "metrics.sqlite"

    with DatabaseManager(db_path) as db:
        db.initialize()

        algo_comp = db.add_component("ppo", "ALGO")
        algo_ver = db.add_component_version(algo_comp, "abc123")
        env_comp = db.add_component("cartpole", "ENV")
        env_ver = db.add_component_version(env_comp, "def456")

        exp_id = db.add_experiment("test-experiment", "unit test fixture")

        runs: list[_RunRecord] = []
        for condition_name, seed in [("oracle", 42), ("mcar", 7)]:
            hyper_id = db.add_hyperparam_config({"condition_name": condition_name, "seed": seed})
            run_id = db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed)

            exec_id = db.add_execution()
            db.link_execution_run(exec_id, run_id)
            executions_root = db_path.parent / "executions" / str(exec_id)
            executions_root.mkdir(parents=True, exist_ok=True)
            db.record_execution_artifacts(
                exec_id,
                str(executions_root),
                metadata={"metrics_db_path": str(metrics_db)},
            )
            db.update_execution_status(exec_id, "COMPLETED")
            runs.append(_RunRecord(run_id=run_id, execution_id=exec_id, seed=seed, condition_name=condition_name))

    return _ExperimentFixture(runs=runs, metrics_db=metrics_db)


def _write_metrics_db(metrics_db: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write rows into a minimal metrics SQLite DB."""
    metrics_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(metrics_db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "metric_name TEXT NOT NULL, "
            "value REAL NOT NULL, "
            "global_step INTEGER NOT NULL, "
            "seed_id INTEGER NOT NULL, "
            "experiment_id INTEGER NOT NULL, "
            "run_id INTEGER NOT NULL, "
            "execution_id INTEGER NOT NULL"
            ")"
        )
        conn.executemany(
            "INSERT INTO metrics(metric_name, value, global_step, seed_id, experiment_id, run_id, execution_id) "
            "VALUES (:metric_name, :value, :global_step, :seed_id, :experiment_id, :run_id, :execution_id)",
            rows,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# resolve_run_artifacts
# ---------------------------------------------------------------------------


class TestResolveRunArtifacts:
    def test_returns_one_row_per_completed_run(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        _bootstrap_experiments_db(db_path)

        df = resolve_run_artifacts(db_path, "test-experiment")

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert set(df.columns) == {"condition_name", "seed", "run_id", "execution_id", "metrics_db"}

    def test_condition_names_round_trip(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        _bootstrap_experiments_db(db_path)

        df = resolve_run_artifacts(db_path, "test-experiment")

        assert set(df["condition_name"].to_list()) == {"oracle", "mcar"}

    def test_seeds_round_trip(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        _bootstrap_experiments_db(db_path)

        df = resolve_run_artifacts(db_path, "test-experiment")

        assert set(df["seed"].to_list()) == {42, 7}

    def test_metrics_db_path_recorded(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        fixture = _bootstrap_experiments_db(db_path)

        df = resolve_run_artifacts(db_path, "test-experiment")

        assert all(p == str(fixture.metrics_db) for p in df["metrics_db"].to_list())

    def test_raises_for_unknown_slug(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        _bootstrap_experiments_db(db_path)

        with pytest.raises(ValueError, match="Unknown experiment slug"):
            resolve_run_artifacts(db_path, "does-not-exist")

    def test_empty_when_no_completed_executions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with DatabaseManager(db_path) as db:
            db.initialize()
            algo_comp = db.add_component("ppo", "ALGO")
            algo_ver = db.add_component_version(algo_comp, "abc123")
            env_comp = db.add_component("cartpole", "ENV")
            env_ver = db.add_component_version(env_comp, "def456")
            exp_id = db.add_experiment("pending-experiment")
            hyper_id = db.add_hyperparam_config({"condition_name": "oracle", "seed": 0})
            db.add_run(exp_id, algo_ver, env_ver, hyper_id, 0)

        df = resolve_run_artifacts(db_path, "pending-experiment")

        assert df.is_empty()


# ---------------------------------------------------------------------------
# load_experiment_metrics
# ---------------------------------------------------------------------------


class TestLoadExperimentMetrics:
    def _setup(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "experiments.sqlite"
        fixture = _bootstrap_experiments_db(db_path)

        metric_rows: list[dict[str, float | int | str]] = []
        for run in fixture.runs:
            for step in [0, 5, 10]:
                metric_rows.append(
                    {
                        "metric_name": "returned_episode_returns",
                        "value": float(step + 1),
                        "global_step": step,
                        "seed_id": run.seed,
                        "experiment_id": 1,
                        "run_id": run.run_id,
                        "execution_id": run.execution_id,
                    }
                )
                metric_rows.append(
                    {
                        "metric_name": "eval_returns",
                        "value": float(step + 2),
                        "global_step": step,
                        "seed_id": run.seed,
                        "experiment_id": 1,
                        "run_id": run.run_id,
                        "execution_id": run.execution_id,
                    }
                )

        _write_metrics_db(fixture.metrics_db, metric_rows)
        return db_path

    def test_returns_tidy_dataframe(self, tmp_path: Path) -> None:
        db_path = self._setup(tmp_path)

        df = load_experiment_metrics(experiments_db=db_path, slug="test-experiment")

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"condition_name", "seed", "step", "metric", "value"}

    def test_contains_all_conditions_and_metrics(self, tmp_path: Path) -> None:
        db_path = self._setup(tmp_path)

        df = load_experiment_metrics(experiments_db=db_path, slug="test-experiment")

        assert set(df["condition_name"].unique().to_list()) == {"oracle", "mcar"}
        assert set(df["metric"].unique().to_list()) == {"returned_episode_returns", "eval_returns"}

    def test_filters_to_requested_metrics(self, tmp_path: Path) -> None:
        db_path = self._setup(tmp_path)

        df = load_experiment_metrics(
            experiments_db=db_path,
            slug="test-experiment",
            metrics=["returned_episode_returns"],
        )

        assert set(df["metric"].unique().to_list()) == {"returned_episode_returns"}

    def test_sorted_by_condition_seed_step_metric(self, tmp_path: Path) -> None:
        db_path = self._setup(tmp_path)

        df = load_experiment_metrics(experiments_db=db_path, slug="test-experiment")

        assert df.equals(df.sort(["condition_name", "seed", "step", "metric"]))

    def test_empty_when_no_completed_executions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "experiments.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with DatabaseManager(db_path) as db:
            db.initialize()
            db.add_experiment("empty-experiment")

        df = load_experiment_metrics(experiments_db=db_path, slug="empty-experiment")

        assert df.is_empty()
        assert set(df.columns) == {"condition_name", "seed", "step", "metric", "value"}

    def test_accepts_pre_resolved_artifacts(self, tmp_path: Path) -> None:
        db_path = self._setup(tmp_path)
        artifacts = resolve_run_artifacts(db_path, "test-experiment")

        df = load_experiment_metrics(
            experiments_db=db_path,
            slug="test-experiment",
            run_artifacts=artifacts,
            metrics=["returned_episode_returns"],
        )

        assert not df.is_empty()
        assert "condition_name" in df.columns

    def test_empty_metrics_list_returns_empty_frame(self, tmp_path: Path) -> None:
        db_path = self._setup(tmp_path)

        df = load_experiment_metrics(experiments_db=db_path, slug="test-experiment", metrics=[])

        assert df.is_empty()
        assert set(df.columns) == {"condition_name", "seed", "step", "metric", "value"}
