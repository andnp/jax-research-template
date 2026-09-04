"""Small tests for research_analysis.experiment — load_experiment_metrics and resolve_run_artifacts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from experiment_definition.db import DatabaseManager
from research_analysis.experiment import load_experiment_metrics, resolve_run_artifacts
from research_analysis.schema import AnalysisSchema

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def test_analysis_schema_maps_named_dimensions() -> None:
    """Map project metadata keys onto canonical analysis dimensions."""
    schema = AnalysisSchema(condition_key="algorithm", environment_key="benchmark")

    assert schema.condition_name({"algorithm": "td3", "benchmark": "chlorine"}) == "td3"
    assert schema.environment_name({"algorithm": "td3", "benchmark": "chlorine"}) == "chlorine"


def test_analysis_schema_rejects_missing_dimension() -> None:
    """Reject incomplete metadata instead of creating an empty condition."""
    schema = AnalysisSchema(condition_key="algorithm", environment_key="benchmark")

    with pytest.raises(ValueError, match="Missing condition metadata key 'algorithm'"):
        schema.condition_name({"benchmark": "chlorine"})


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


def _bootstrap_mapped_experiments_db(db_path: Path) -> tuple[Path, list[tuple[int, int, int]]]:
    """Create a synthetic experiment using algorithm and benchmark metadata."""
    metrics_db = db_path.parent / "metrics.sqlite"
    with DatabaseManager(db_path) as db:
        db.initialize()
        algo_comp = db.add_component("td3", "ALGO")
        algo_ver = db.add_component_version(algo_comp, "abc123")
        env_comp = db.add_component("chlorine", "ENV")
        env_ver = db.add_component_version(env_comp, "def456")
        exp_id = db.add_experiment("mapped-experiment")

        run_records: list[tuple[int, int, int]] = []
        for seed, algorithm in enumerate(("td3", "sac")):
            hyper_id = db.add_hyperparam_config(
                {"algorithm": algorithm, "benchmark": "chlorine", "seed": seed}
            )
            run_id = db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed)
            execution_id = db.add_execution()
            db.link_execution_run(execution_id, run_id)
            execution_root = db_path.parent / "executions" / str(execution_id)
            execution_root.mkdir(parents=True, exist_ok=True)
            db.record_execution_artifacts(
                execution_id,
                str(execution_root),
                metadata={"metrics_db_path": str(metrics_db)},
            )
            db.update_execution_status(execution_id, "COMPLETED")
            run_records.append((run_id, execution_id, seed))

    return metrics_db, run_records


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
# resolve_run_artifacts — latest_component_versions
# ---------------------------------------------------------------------------


def _bootstrap_versioned_experiment(db_path: Path) -> tuple[int, int, int, int]:
    """Create one experiment with two ALGO component versions, one completed run each.

    Returns (older_algo_version_id, newer_algo_version_id, older_run_id, newer_run_id).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with DatabaseManager(db_path) as db:
        db.initialize()
        algo_comp = db.add_component("ppo", "ALGO")
        older_algo_ver = db.add_component_version(algo_comp, "commit-buggy")
        newer_algo_ver = db.add_component_version(algo_comp, "commit-fixed")
        env_comp = db.add_component("cartpole", "ENV")
        env_ver = db.add_component_version(env_comp, "def456")
        exp_id = db.add_experiment("versioned-experiment")

        older_hyper = db.add_hyperparam_config({"condition_name": "buggy", "seed": 0})
        newer_hyper = db.add_hyperparam_config({"condition_name": "fixed", "seed": 0})
        older_run = db.add_run(exp_id, older_algo_ver, env_ver, older_hyper, 0)
        newer_run = db.add_run(exp_id, newer_algo_ver, env_ver, newer_hyper, 0)

        for run_id in (older_run, newer_run):
            exec_id = db.add_execution()
            db.link_execution_run(exec_id, run_id)
            db.record_execution_artifacts(exec_id, str(db_path.parent / "executions" / str(exec_id)))
            db.update_execution_status(exec_id, "COMPLETED")

    return older_algo_ver, newer_algo_ver, older_run, newer_run


class TestLatestComponentVersions:
    def test_default_filters_to_newest_algo_version(self, tmp_path: Path) -> None:
        """Pooling every code version by default is the bug; the default must return only the newest."""
        db_path = tmp_path / "experiments.sqlite"
        older_ver, newer_ver, older_run, newer_run = _bootstrap_versioned_experiment(db_path)
        assert newer_ver > older_ver  # fixture premise: the intended "newest" really is the max

        df = resolve_run_artifacts(db_path, "versioned-experiment")

        assert set(df["run_id"].to_list()) == {newer_run}
        assert older_run not in df["run_id"].to_list()

    def test_opt_out_restores_pooling_across_versions(self, tmp_path: Path) -> None:
        """Deliberate cross-version comparison (e.g. diagnosing this exact bug) must remain possible."""
        db_path = tmp_path / "experiments.sqlite"
        _, _, older_run, newer_run = _bootstrap_versioned_experiment(db_path)

        df = resolve_run_artifacts(db_path, "versioned-experiment", latest_component_versions=False)

        assert set(df["run_id"].to_list()) == {older_run, newer_run}

    def test_null_algo_version_id_still_returns_rows(self, tmp_path: Path) -> None:
        """A multi-ALGO-component study stores NULL algo_version_id for every run; the default must match NULL, not drop the rows."""
        db_path = tmp_path / "experiments.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with DatabaseManager(db_path) as db:
            db.initialize()
            env_comp = db.add_component("cartpole", "ENV")
            env_ver = db.add_component_version(env_comp, "def456")
            exp_id = db.add_experiment("null-version-experiment")
            hyper_id = db.add_hyperparam_config({"condition_name": "multi", "seed": 0})
            with db.conn:
                cur = db.conn.execute(
                    "INSERT INTO Runs(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation) "
                    "VALUES (?, NULL, ?, ?, ?, NULL)",
                    (exp_id, env_ver, hyper_id, 0),
                )
            assert cur.lastrowid is not None
            run_id = int(cur.lastrowid)

            exec_id = db.add_execution()
            db.link_execution_run(exec_id, run_id)
            db.record_execution_artifacts(exec_id, str(db_path.parent / "executions" / str(exec_id)))
            db.update_execution_status(exec_id, "COMPLETED")

        df = resolve_run_artifacts(db_path, "null-version-experiment")

        assert df["run_id"].to_list() == [run_id]


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

    def test_maps_project_metadata_without_condition_collapse(self, tmp_path: Path) -> None:
        """Preserve algorithm and benchmark dimensions through canonical loading."""
        db_path = tmp_path / "experiments.sqlite"
        metrics_db, run_records = _bootstrap_mapped_experiments_db(db_path)
        _write_metrics_db(
            metrics_db,
            [
                {
                    "metric_name": "reward",
                    "value": float(seed + 1),
                    "global_step": 0,
                    "seed_id": seed,
                    "experiment_id": 1,
                    "run_id": run_id,
                    "execution_id": execution_id,
                }
                for run_id, execution_id, seed in run_records
            ],
        )

        df = load_experiment_metrics(
            experiments_db=db_path,
            slug="mapped-experiment",
            schema=AnalysisSchema(condition_key="algorithm", environment_key="benchmark"),
        )

        assert set(df["condition_name"].to_list()) == {"td3", "sac"}
        assert set(df["env_name"].to_list()) == {"chlorine"}
        assert set(df["metric"].to_list()) == {"reward"}
