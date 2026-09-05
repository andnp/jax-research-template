"""Contract tests for the canonical experiment dataset boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

import pytest
from experiment_definition.db import DatabaseManager
from research_analysis.dataset import (
    CohortSelection,
    ComponentVersionRef,
    DatasetLoadError,
    DiagnosticCode,
    ExperimentDimensions,
    SweepAxis,
    SweepValueType,
    VersionSelection,
    load_experiment_dataset,
)


class DatasetFixture(TypedDict):
    db_path: Path
    metrics_path: Path
    algo_version: int
    env_version: int


def _write_metrics(path: Path, rows: list[tuple[int, int, str, int, float]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE metrics (run_id INTEGER, execution_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
        )
        connection.executemany(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def _create_experiment(tmp_path: Path, *, versions: tuple[str, str] = ("algo-v1", "env-v1")) -> DatasetFixture:
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("dataset-test")
        algo_component = database.add_component("ppo", "ALGO")
        algo_version = database.add_component_version(algo_component, versions[0])
        env_component = database.add_component("cartpole", "ENV")
        env_version = database.add_component_version(env_component, versions[1])
        rows: list[tuple[int, int, int]] = []
        for seed, condition in enumerate(("baseline", "candidate")):
            hyper_id = database.add_hyperparam_config(
                {"condition_name": condition, "learning_rate": 0.1 + seed, "seed": seed}
            )
            run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, seed)
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.record_execution_artifacts(
                execution_id,
                str(tmp_path / "execution"),
                metadata={"metrics_db_path": str(metrics_path)},
            )
            database.update_execution_status(execution_id, "COMPLETED")
            rows.append((run_id, execution_id, seed))
    _write_metrics(
        metrics_path,
        [(run_id, execution_id, "reward", 0, float(seed)) for run_id, execution_id, seed in rows],
    )
    return {
        "db_path": db_path,
        "metrics_path": metrics_path,
        "algo_version": algo_version,
        "env_version": env_version,
    }


def test_load_experiment_dataset_preserves_typed_runs_and_metrics(tmp_path: Path) -> None:
    """Load canonical dimensions, sweep axes, provenance, and metric identity."""
    fixture = _create_experiment(tmp_path)

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        dimensions=ExperimentDimensions(
            sweep_axes=(SweepAxis("learning_rate", SweepValueType.FLOAT),),
        ),
    )

    assert dataset.runs["condition_name"].to_list() == ["baseline", "candidate"]
    assert dataset.runs["learning_rate"].dtype.is_float()
    assert dataset.metrics.select("metric").unique().to_series().to_list() == ["reward"]
    assert dataset.runs["algo_component_name"].to_list() == ["ppo", "ppo"]
    assert dataset.runs["algo_version_id"].to_list() == [fixture["algo_version"], fixture["algo_version"]]


def test_latest_completed_selects_one_coherent_version_set(tmp_path: Path) -> None:
    """Select a completed algorithm/environment pair without mixing version maxima."""
    fixture = _create_experiment(tmp_path)
    db_path = fixture["db_path"]
    with DatabaseManager(db_path) as database:
        experiment = database.get_experiment("dataset-test")
        assert experiment is not None
        component = database.get_component("ppo")
        assert component is not None
        newer_algo = database.add_component_version(component.id, "algo-v2")
        hyper_id = database.add_hyperparam_config({"condition_name": "new", "learning_rate": 0.3, "seed": 9})
        run_id = database.add_run(experiment.id, newer_algo, fixture["env_version"], hyper_id, 9)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id,
            str(tmp_path / "execution-new"),
            metadata={"metrics_db_path": str(fixture["metrics_path"])},
        )
        database.update_execution_status(execution_id, "COMPLETED")
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?)", (run_id, execution_id, "reward", 0, 9.0))
        connection.commit()

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test")

    assert dataset.runs["algo_version_id"].unique().to_list() == [newer_algo]
    assert not any(item.code is DiagnosticCode.AMBIGUOUS_COHORT for item in dataset.diagnostics)


def test_exact_versions_require_qualified_component_identity(tmp_path: Path) -> None:
    """Reject unqualified version numbers and accept a component hash selector."""
    fixture = _create_experiment(tmp_path)

    with pytest.raises(ValueError, match="exactly one version_id or code_snapshot_hash"):
        ComponentVersionRef("ppo", "ALGO")

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        version_selection=VersionSelection.exact(
            (
                ComponentVersionRef("ppo", "ALGO", code_snapshot_hash="algo-v1"),
                ComponentVersionRef("cartpole", "ENV", code_snapshot_hash="env-v1"),
            )
        ),
    )

    assert dataset.runs.height == 2


def test_missing_recorded_metric_path_fails_without_root_fallback(tmp_path: Path) -> None:
    """Fail strict loading when metadata omits the metric path even if root has a database."""
    db_path = tmp_path / "experiments.sqlite"
    root = tmp_path / "execution"
    root.mkdir()
    _write_metrics(root / "metrics.sqlite", [])
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("missing-artifact")
        algo_component = database.add_component("ppo", "ALGO")
        algo_version = database.add_component_version(algo_component, "algo")
        env_component = database.add_component("cartpole", "ENV")
        env_version = database.add_component_version(env_component, "env")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "seed": 0})
        run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, 0)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(execution_id, str(root))
        database.update_execution_status(execution_id, "COMPLETED")

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(db_path, "missing-artifact")

    assert error.value.diagnostics[0].code is DiagnosticCode.MISSING_METRIC_ARTIFACT


def test_partial_mode_returns_valid_rows_and_diagnostics(tmp_path: Path) -> None:
    """Return usable rows while retaining missing-artifact diagnostics in partial mode."""
    fixture = _create_experiment(tmp_path)
    with DatabaseManager(fixture["db_path"]) as database:
        experiment = database.get_experiment("dataset-test")
        assert experiment is not None
        hyper_id = database.add_hyperparam_config({"condition_name": "missing", "learning_rate": 0.9, "seed": 9})
        run_id = database.add_run(experiment.id, fixture["algo_version"], fixture["env_version"], hyper_id, 9)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(execution_id, str(tmp_path / "missing"), metadata={"metrics_db_path": str(tmp_path / "absent.sqlite")})
        database.update_execution_status(execution_id, "COMPLETED")

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test", partial=True)

    assert dataset.runs.height == 3
    assert any(item.code is DiagnosticCode.MISSING_METRIC_ARTIFACT for item in dataset.diagnostics)


def test_cohort_selection_filters_declared_dimensions(tmp_path: Path) -> None:
    """Select one declared condition without collapsing other run identities."""
    fixture = _create_experiment(tmp_path)

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        cohort=CohortSelection(condition_name="candidate"),
    )

    assert dataset.runs["condition_name"].to_list() == ["candidate"]
