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
    ExperimentDimensions,
    SweepAxis,
    SweepValueType,
    VersionSelection,
    load_experiment_dataset,
)
from research_analysis.diagnostics.types import DatasetLoadError, DiagnosticCode


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


def test_environment_dimension_populates_row_and_provenance(tmp_path: Path) -> None:
    """Environment dimension resolves to env_name and stamps env component provenance."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("environment-dimension-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_component = database.add_component("cartpole", "ENV")
        env_version = database.add_component_version(env_component, "env-v1")
        hyper_id = database.add_hyperparam_config(
            {"condition_name": "baseline", "env_id": "cartpole-v1", "seed": 0}
        )
        run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, 0)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
        )
        database.update_execution_status(execution_id, "COMPLETED")
    _write_metrics(metrics_path, [(run_id, execution_id, "reward", 0, 1.0)])

    dataset = load_experiment_dataset(
        db_path,
        "environment-dimension-test",
        dimensions=ExperimentDimensions(environment="env_id"),
    )

    assert dataset.runs["env_name"].to_list() == ["cartpole-v1"]
    assert dataset.runs["env_component_name"].to_list() == ["cartpole"]
    assert dataset.runs["env_version_id"].to_list() == [env_version]


def test_missing_environment_dimension_fails(tmp_path: Path) -> None:
    """An unset environment key fails instead of silently omitting env_name."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("missing-environment-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "seed": 0})
        run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, 0)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
        )
        database.update_execution_status(execution_id, "COMPLETED")
    _write_metrics(metrics_path, [(run_id, execution_id, "reward", 0, 1.0)])

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(
            db_path,
            "missing-environment-test",
            dimensions=ExperimentDimensions(environment="env_id"),
        )

    assert error.value.diagnostics[0].code is DiagnosticCode.MISSING_DIMENSION


def test_missing_applicable_sweep_axis_fails(tmp_path: Path) -> None:
    """An applicable axis without a value fails instead of becoming nullable."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("required-axis-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "seed": 0})
        run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, 0)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
        )
        database.update_execution_status(execution_id, "COMPLETED")
    _write_metrics(metrics_path, [(run_id, execution_id, "reward", 0, 1.0)])

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(
            db_path,
            "required-axis-test",
            dimensions=ExperimentDimensions(
                sweep_axes=(
                    SweepAxis("learning_rate", SweepValueType.FLOAT, when={"condition_name": "baseline"}),
                ),
            ),
        )

    assert error.value.diagnostics[0].code is DiagnosticCode.MISSING_DIMENSION


def test_inapplicable_sweep_axis_is_nullable_instead_of_excluding(tmp_path: Path) -> None:
    """An inactive axis is null while an applicable axis retains its typed value."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("optional-axis-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        rows = []
        no_eta_id = database.add_hyperparam_config({"condition_name": "shared", "lr": 0.1, "seed": 0})
        run_no_eta = database.add_run(experiment_id, algo_version, env_version, no_eta_id, 0)
        with_eta_id = database.add_hyperparam_config({"condition_name": "adversarial", "lr": 0.1, "eta": 5.0, "seed": 1})
        run_with_eta = database.add_run(experiment_id, algo_version, env_version, with_eta_id, 1)
        for run_id, seed in ((run_no_eta, 0), (run_with_eta, 1)):
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.record_execution_artifacts(
                execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
            )
            database.update_execution_status(execution_id, "COMPLETED")
            rows.append((run_id, execution_id, "reward", 0, float(seed)))
    _write_metrics(metrics_path, rows)

    dataset = load_experiment_dataset(
        db_path,
        "optional-axis-test",
        dimensions=ExperimentDimensions(
            sweep_axes=(
                SweepAxis("lr", SweepValueType.FLOAT),
                SweepAxis("eta", SweepValueType.FLOAT, when={"condition_name": "adversarial"}),
            ),
        ),
    )

    assert dataset.runs.height == 2
    by_condition = dict(zip(dataset.runs["condition_name"], dataset.runs["eta"], strict=True))
    assert by_condition["shared"] is None
    assert by_condition["adversarial"] == 5.0
    shared_only = load_experiment_dataset(
        db_path,
        "optional-axis-test",
        dimensions=ExperimentDimensions(
            sweep_axes=(
                SweepAxis("lr", SweepValueType.FLOAT),
                SweepAxis("eta", SweepValueType.FLOAT, when={"condition_name": "adversarial"}),
            ),
        ),
        cohort=CohortSelection(condition_name="shared"),
    )
    assert shared_only.runs["eta"].dtype.is_float()


def test_inapplicable_sweep_axis_with_value_fails(tmp_path: Path) -> None:
    """Reject an axis value that appears outside its declared condition."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("invalid-conditional-axis-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        hyper_id = database.add_hyperparam_config(
            {"condition_name": "shared", "lr": 0.1, "eta": 5.0, "seed": 0}
        )
        run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, 0)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
        )
        database.update_execution_status(execution_id, "COMPLETED")
    _write_metrics(metrics_path, [(run_id, execution_id, "reward", 0, 1.0)])

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(
            db_path,
            "invalid-conditional-axis-test",
            dimensions=ExperimentDimensions(
                sweep_axes=(
                    SweepAxis("eta", SweepValueType.FLOAT, when={"condition_name": "adversarial"}),
                ),
            ),
        )

    assert error.value.diagnostics[0].code is DiagnosticCode.INVALID_DIMENSION


def test_missing_applicability_key_fails(tmp_path: Path) -> None:
    """Reject an applicability rule that cannot be evaluated from run metadata."""
    fixture = _create_experiment(tmp_path)

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(
            fixture["db_path"],
            "dataset-test",
            dimensions=ExperimentDimensions(
                sweep_axes=(SweepAxis("learning_rate", SweepValueType.FLOAT, when={"unknown": "value"}),),
            ),
        )

    assert error.value.diagnostics[0].code is DiagnosticCode.INVALID_DIMENSION


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
    assert any(item.code is DiagnosticCode.EXCLUDED_VERSION for item in dataset.diagnostics)
    provenance_versions = {item.version_id for item in dataset.provenance}
    assert newer_algo in provenance_versions
    assert fixture["algo_version"] not in provenance_versions


def test_version_selection_all_includes_every_completed_version_set(tmp_path: Path) -> None:
    """VersionSelection.all() keeps every completed version set without excluding any."""
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

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test", version_selection=VersionSelection.all())

    assert dataset.runs.height == 3
    assert not any(item.code is DiagnosticCode.EXCLUDED_VERSION for item in dataset.diagnostics)
    provenance_versions = {item.version_id for item in dataset.provenance}
    assert {newer_algo, fixture["algo_version"]} <= provenance_versions


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


def test_exact_version_selection_without_matches_fails(tmp_path: Path) -> None:
    """No completed run qualifies under an exact selector that matches nothing."""
    fixture = _create_experiment(tmp_path)

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(
            fixture["db_path"],
            "dataset-test",
            version_selection=VersionSelection.exact(
                (ComponentVersionRef("ppo", "ALGO", code_snapshot_hash="algo-v-nonexistent"),)
            ),
        )

    assert error.value.diagnostics[0].code is DiagnosticCode.INVALID_VERSION_SELECTION


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


def test_cohort_selection_filters_by_env_name_and_sweep_value(tmp_path: Path) -> None:
    """Cohort selection narrows to a run matching both env_name and a declared sweep value."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("cohort-env-sweep-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        rows = []
        for seed, (env_id, lr) in enumerate((("cartpole-v1", 0.1), ("pendulum-v1", 0.2))):
            hyper_id = database.add_hyperparam_config(
                {"condition_name": "shared", "env_id": env_id, "lr": lr, "seed": seed}
            )
            run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, seed)
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.record_execution_artifacts(
                execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(metrics_path)}
            )
            database.update_execution_status(execution_id, "COMPLETED")
            rows.append((run_id, execution_id, "reward", 0, float(seed)))
    _write_metrics(metrics_path, rows)

    dataset = load_experiment_dataset(
        db_path,
        "cohort-env-sweep-test",
        dimensions=ExperimentDimensions(
            environment="env_id",
            sweep_axes=(SweepAxis("lr", SweepValueType.FLOAT),),
        ),
        cohort=CohortSelection(env_name="cartpole-v1", sweep_values=(("lr", 0.1),)),
    )

    assert dataset.runs.height == 1
    assert dataset.runs["env_name"].to_list() == ["cartpole-v1"]


def test_cohort_selection_filters_declared_dimensions(tmp_path: Path) -> None:
    """Select one declared condition without collapsing other run identities."""
    fixture = _create_experiment(tmp_path)

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        cohort=CohortSelection(condition_name="candidate"),
    )

    assert dataset.runs["condition_name"].to_list() == ["candidate"]
