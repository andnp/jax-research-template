"""Contract tests for the canonical experiment dataset boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

import pytest
from experiment_definition.db import DatabaseManager
from research_analysis.dataset.dimensions import UnresolvedDimension, resolve_dimensions
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import (
    CohortSelection,
    ComponentVersionRef,
    ExclusionReason,
    ExperimentDimensions,
    SweepAxis,
    SweepValueType,
    VersionSelection,
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


def _add_incomplete_run(db_path: Path, fixture: DatasetFixture, seed: int, statuses: tuple[str, ...]) -> int:
    with DatabaseManager(db_path) as database:
        experiment = database.get_experiment("dataset-test")
        assert experiment is not None
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "learning_rate": 0.1, "seed": seed})
        run_id = database.add_run(experiment.id, fixture["algo_version"], fixture["env_version"], hyper_id, seed)
        for status in statuses:
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            if status != "PENDING":
                database.update_execution_status(execution_id, status)
    return run_id


def test_incomplete_runs_are_diagnosed_as_failed_or_pending(tmp_path: Path) -> None:
    """Runs without a completed execution are reported, separating crashes from unfinished work.

    A failed attempt followed by a queued retry is still pending, not failed.
    """
    fixture = _create_experiment(tmp_path)
    failed = _add_incomplete_run(fixture["db_path"], fixture, 10, ("FAILED",))
    retried = _add_incomplete_run(fixture["db_path"], fixture, 11, ("FAILED", "PENDING"))
    unexecuted = _add_incomplete_run(fixture["db_path"], fixture, 12, ())

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test")

    codes = {item.run_id: item.code for item in dataset.diagnostics}
    assert codes == {
        failed: DiagnosticCode.RUN_FAILED,
        retried: DiagnosticCode.RUN_PENDING,
        unexecuted: DiagnosticCode.RUN_PENDING,
    }
    assert dataset.runs.height == 2


def test_missing_hyperparameter_config_fails_loading(tmp_path: Path) -> None:
    """A run whose hyperparameter config is gone cannot be placed in a cell, so strict loading fails."""
    fixture = _create_experiment(tmp_path)
    with sqlite3.connect(fixture["db_path"]) as connection:
        connection.execute("UPDATE Runs SET hyper_id = 999 WHERE seed = 0")

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(fixture["db_path"], "dataset-test")

    assert [item.code for item in error.value.diagnostics] == [DiagnosticCode.MISSING_HYPER_CONFIG]


def test_undeclared_hyperparameter_variation_within_a_cell_fails_loading(tmp_path: Path) -> None:
    """Runs of one condition that differ in an undeclared hyperparameter are ambiguous, not pooled."""
    fixture = _create_experiment(tmp_path)
    with DatabaseManager(fixture["db_path"]) as database:
        experiment = database.get_experiment("dataset-test")
        assert experiment is not None
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "learning_rate": 0.5, "seed": 5})
        run_id = database.add_run(experiment.id, fixture["algo_version"], fixture["env_version"], hyper_id, 5)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id, str(tmp_path / "execution"), metadata={"metrics_db_path": str(fixture["metrics_path"])}
        )
        database.update_execution_status(execution_id, "COMPLETED")
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?)", (run_id, execution_id, "reward", 0, 2.0))

    with pytest.raises(DatasetLoadError, match=r"undeclared hyperparameters \['learning_rate'\] differ") as error:
        load_experiment_dataset(fixture["db_path"], "dataset-test")

    assert {item.code for item in error.value.diagnostics} == {DiagnosticCode.AMBIGUOUS_COHORT}


def test_accounting_records_every_planned_run_per_cell(tmp_path: Path) -> None:
    """Every planned run appears in its cell's accounting as completed, failed, pending, or excluded with a reason.

    Diverged runs count as completed and are tallied per metric; runs with no resolvable cell are unassigned.
    """
    fixture = _create_experiment(tmp_path)
    _add_incomplete_run(fixture["db_path"], fixture, 10, ("FAILED",))
    _add_incomplete_run(fixture["db_path"], fixture, 11, ())
    with DatabaseManager(fixture["db_path"]) as database:
        experiment = database.get_experiment("dataset-test")
        component = database.get_component("ppo")
        assert experiment is not None
        assert component is not None
        newer_algo = database.add_component_version(component.id, "algo-v2")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "learning_rate": 0.1, "seed": 12})
        newer_run = database.add_run(experiment.id, newer_algo, fixture["env_version"], hyper_id, 12)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, newer_run)
        database.update_execution_status(execution_id, "COMPLETED")
        unlabelled_id = database.add_hyperparam_config({"algorithm": "ppo", "seed": 13})
        database.add_run(experiment.id, fixture["algo_version"], fixture["env_version"], unlabelled_id, 13)
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("UPDATE metrics SET value = NULL WHERE value = 1.0")

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        version_selection=VersionSelection.exact(
            (ComponentVersionRef.by_id("ppo", "ALGO", fixture["algo_version"]),)
        ),
        partial=True,
    )

    cells = {dict(item.cell)["condition_name"]: item for item in dataset.accounting.cells}
    baseline, candidate = cells["baseline"], cells["candidate"]
    assert (baseline.planned, baseline.completed, baseline.failed, baseline.pending) == (4, 1, 1, 1)
    assert baseline.excluded == {ExclusionReason.VERSION_NOT_SELECTED: 1}
    assert baseline.diverged == {}
    assert (candidate.planned, candidate.completed, candidate.diverged) == (1, 1, {"reward": 1})
    assert dataset.accounting.unassigned == {ExclusionReason.UNRESOLVED_DIMENSION: 1}


def test_exact_selection_without_matches_counts_every_run_as_not_selected(tmp_path: Path) -> None:
    """An exact selection that matches nothing must not count the unselected completed runs as completed."""
    fixture = _create_experiment(tmp_path)
    _add_incomplete_run(fixture["db_path"], fixture, 10, ())

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        version_selection=VersionSelection.exact((ComponentVersionRef.by_id("ppo", "ALGO", 999),)),
        partial=True,
    )

    assert [(cell.completed, cell.pending, dict(cell.excluded)) for cell in dataset.accounting.cells] == [
        (0, 0, {ExclusionReason.VERSION_NOT_SELECTED: 2}),
        (0, 0, {ExclusionReason.VERSION_NOT_SELECTED: 1}),
    ]
    assert [item.code for item in dataset.diagnostics] == [
        DiagnosticCode.INVALID_VERSION_SELECTION,
        DiagnosticCode.EXCLUDED_VERSION,
    ]


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
    run = dataset.runs.row(0, named=True)
    run_artifacts = dataset.artifacts[(run["run_id"], run["execution_id"])]
    assert run_artifacts.root_path == tmp_path / "execution"
    assert "metrics_db_path" not in run_artifacts.metadata
    assert "metrics_db_path" not in dataset.runs.columns


def test_stored_null_metric_values_load_as_nan(tmp_path: Path) -> None:
    """The metric store records NaN as NULL; the dataset must surface it as NaN, not a missing value."""
    fixture = _create_experiment(tmp_path)
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("UPDATE metrics SET value = NULL")

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test")

    assert dataset.metrics["value"].is_nan().to_list() == [True, True]


def test_diverged_run_is_kept_and_flagged(tmp_path: Path) -> None:
    """A run whose metric went infinite stays in the dataset, is flagged, and is reported per metric."""
    fixture = _create_experiment(tmp_path)
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("UPDATE metrics SET value = ? WHERE value = 1.0", (float("-inf"),))

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test")

    flags = dict(zip(dataset.runs["condition_name"], dataset.runs["non_finite_metrics"].to_list(), strict=True))
    assert flags == {"baseline": [], "candidate": ["reward"]}
    assert [(item.code, item.run_id) for item in dataset.diagnostics] == [
        (DiagnosticCode.NON_FINITE_METRIC, dataset.runs["run_id"][1])
    ]


@pytest.mark.parametrize("corruption", ["directory", "not_sqlite", "no_metrics_table"])
def test_unreadable_metrics_database_is_a_malformed_artifact(tmp_path: Path, corruption: str) -> None:
    """A recorded metrics path that is not a readable metrics database yields a structured diagnostic, not a crash."""
    fixture = _create_experiment(tmp_path)
    metrics_path = fixture["metrics_path"]
    metrics_path.unlink()
    if corruption == "directory":
        metrics_path.mkdir()
    elif corruption == "not_sqlite":
        metrics_path.write_bytes(b"not a database" * 100)
    else:
        sqlite3.connect(metrics_path).close()

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(fixture["db_path"], "dataset-test")
    partial = load_experiment_dataset(fixture["db_path"], "dataset-test", partial=True)

    assert {item.code for item in error.value.diagnostics} == {DiagnosticCode.MALFORMED_ARTIFACT}
    assert partial.metrics.height == 0
    assert [dict(cell.excluded) for cell in partial.accounting.cells] == [{ExclusionReason.MALFORMED_ARTIFACT: 1}] * 2


def test_run_without_metric_rows_fails_loading(tmp_path: Path) -> None:
    """A run absent from a metrics database that holds other runs' rows is reported, not dropped."""
    fixture = _create_experiment(tmp_path)
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("DELETE FROM metrics WHERE value = 1.0")

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(fixture["db_path"], "dataset-test")

    assert [item.code for item in error.value.diagnostics] == [DiagnosticCode.MISSING_METRIC_ARTIFACT]


def test_empty_dataset_preserves_run_schema_without_storage_columns(tmp_path: Path) -> None:
    """Empty and populated datasets expose the same analysis columns."""
    fixture = _create_experiment(tmp_path)
    dimensions = ExperimentDimensions(
        sweep_axes=(SweepAxis("learning_rate", SweepValueType.FLOAT),),
    )

    populated = load_experiment_dataset(fixture["db_path"], "dataset-test", dimensions=dimensions)
    empty = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        dimensions=dimensions,
        cohort=CohortSelection(condition_name="missing"),
    )

    assert empty.runs.schema == populated.runs.schema
    assert "metrics_db_path" not in empty.runs.columns


def test_invalid_dimension_labels_are_unresolved_without_stringifying_values() -> None:
    """Reject non-string or blank labels instead of coercing them into dimensions."""
    dimensions = ExperimentDimensions()

    for value in (True, 1.5, ""):
        resolved = resolve_dimensions({"condition_name": value}, dimensions)
        assert isinstance(resolved, UnresolvedDimension)
        assert "'condition_name' must be a non-empty string" in resolved.reason


def test_missing_condition_key_explains_the_condition_naming_default() -> None:
    """A run labelled under another key gets a hint that the condition dimension defaults to condition_name."""
    resolved = resolve_dimensions({"algorithm": "ppo", "seed": 0}, ExperimentDimensions())

    assert isinstance(resolved, UnresolvedDimension)
    assert "ExperimentDimensions(condition='algorithm')" in resolved.reason
    assert "['algorithm']" in resolved.reason


def test_invalid_dimension_configuration_fails_at_construction() -> None:
    """Reject empty source keys before dataset loading begins."""
    with pytest.raises(ValueError, match="condition must name"):
        ExperimentDimensions(condition=" ")
    with pytest.raises(ValueError, match="environment must name"):
        ExperimentDimensions(environment="")


def test_dataset_loads_metric_cohorts_larger_than_one_sql_query(tmp_path: Path) -> None:
    """Batch metric queries when one cohort exceeds SQLite's parameter budget."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("large-cohort-test")
        algo_version = database.add_component_version(database.add_component("ppo", "ALGO"), "algo-v1")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")
        metric_rows: list[tuple[int, int, str, int, float]] = []
        for seed in range(451):
            hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "seed": seed})
            run_id = database.add_run(experiment_id, algo_version, env_version, hyper_id, seed)
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.record_execution_artifacts(
                execution_id,
                str(tmp_path / "execution"),
                metadata={"metrics_db_path": str(metrics_path)},
            )
            database.update_execution_status(execution_id, "COMPLETED")
            metric_rows.append((run_id, execution_id, "reward", 0, float(seed)))
    _write_metrics(metrics_path, metric_rows)

    dataset = load_experiment_dataset(db_path, "large-cohort-test")

    assert dataset.runs.height == 451
    assert dataset.metrics.height == 451


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


def test_unsupported_component_provenance_fails_before_environment_projection(tmp_path: Path) -> None:
    """Reject unsupported component roles instead of relabeling them as environments."""
    fixture = _create_experiment(tmp_path)
    with DatabaseManager(fixture["db_path"]) as database:
        experiment = database.get_experiment("dataset-test")
        assert experiment is not None
        wrapper = database.add_component_version(database.add_component("wrapper", "WRAPPER"), "wrapper-v1")
        hyper_id = database.add_hyperparam_config({"condition_name": "wrapper", "seed": 2})
        run_id = database.add_run(experiment.id, fixture["algo_version"], wrapper, hyper_id, 2)
        execution_id = database.add_execution()
        database.link_execution_run(execution_id, run_id)
        database.record_execution_artifacts(
            execution_id,
            str(tmp_path / "execution-wrapper"),
            metadata={"metrics_db_path": str(fixture["metrics_path"])},
        )
        database.update_execution_status(execution_id, "COMPLETED")
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?)", (run_id, execution_id, "reward", 0, 2.0))
        connection.commit()

    with pytest.raises(ValueError, match="unsupported dataset component type"):
        load_experiment_dataset(fixture["db_path"], "dataset-test", version_selection=VersionSelection.all())


def test_missing_environment_dimension_fails_loading(tmp_path: Path) -> None:
    """An unset environment key fails strict loading instead of silently excluding the run."""
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

    with pytest.raises(DatasetLoadError, match="env_name dimension key 'env_id' is missing") as error:
        load_experiment_dataset(
            db_path,
            "missing-environment-test",
            dimensions=ExperimentDimensions(environment="env_id"),
        )

    assert [item.code for item in error.value.diagnostics] == [DiagnosticCode.UNRESOLVED_DIMENSION]


def test_missing_applicable_sweep_axis_fails_loading(tmp_path: Path) -> None:
    """An applicable axis without a value fails strict loading and names the axis."""
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

    with pytest.raises(DatasetLoadError, match="sweep axis 'learning_rate' is missing"):
        load_experiment_dataset(
            db_path,
            "required-axis-test",
            dimensions=ExperimentDimensions(
                sweep_axes=(
                    SweepAxis("learning_rate", SweepValueType.FLOAT, when={"condition_name": "baseline"}),
                ),
            ),
        )


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


def test_inapplicable_sweep_axis_with_value_is_ignored(tmp_path: Path) -> None:
    """Ignore an axis value that appears outside its declared condition."""
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

    dataset = load_experiment_dataset(
        db_path,
        "invalid-conditional-axis-test",
        dimensions=ExperimentDimensions(
            sweep_axes=(
                SweepAxis("eta", SweepValueType.FLOAT, when={"condition_name": "adversarial"}),
            ),
        ),
    )

    assert dataset.runs.height == 1
    assert dataset.runs["eta"].to_list() == [None]
    assert dataset.diagnostics == ()


def test_missing_applicability_key_fails_loading(tmp_path: Path) -> None:
    """An applicability rule that cannot be evaluated from run metadata fails strict loading."""
    fixture = _create_experiment(tmp_path)

    with pytest.raises(DatasetLoadError, match=r"applicability keys \['unknown'\] are missing"):
        load_experiment_dataset(
            fixture["db_path"],
            "dataset-test",
            dimensions=ExperimentDimensions(
                sweep_axes=(SweepAxis("learning_rate", SweepValueType.FLOAT, when={"unknown": "value"}),),
            ),
        )


def test_latest_completed_rejects_cells_orphaned_by_a_partial_rerun(tmp_path: Path) -> None:
    """A newer version set covering only some cells must not silently drop the cells it lacks."""
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

    with pytest.raises(DatasetLoadError) as error:
        load_experiment_dataset(fixture["db_path"], "dataset-test")

    orphaned = [item.message for item in error.value.diagnostics if item.code is DiagnosticCode.AMBIGUOUS_COHORT]
    assert len(orphaned) == 2
    assert any("'baseline'" in message for message in orphaned)
    assert any("'candidate'" in message for message in orphaned)


def test_latest_completed_supersedes_a_fully_rerun_version_set(tmp_path: Path) -> None:
    """When the newer version set covers every cell, older runs are excluded with a non-fatal diagnostic."""
    fixture = _create_experiment(tmp_path)
    metric_rows: list[tuple[int, int, str, int, float]] = []
    with DatabaseManager(fixture["db_path"]) as database:
        experiment = database.get_experiment("dataset-test")
        component = database.get_component("ppo")
        assert experiment is not None
        assert component is not None
        newer_algo = database.add_component_version(component.id, "algo-v2")
        for seed, condition in enumerate(("baseline", "candidate")):
            hyper_id = database.add_hyperparam_config({"condition_name": condition, "learning_rate": 0.1 + seed, "seed": seed})
            run_id = database.add_run(experiment.id, newer_algo, fixture["env_version"], hyper_id, seed)
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.record_execution_artifacts(
                execution_id, str(tmp_path / "execution-new"), metadata={"metrics_db_path": str(fixture["metrics_path"])}
            )
            database.update_execution_status(execution_id, "COMPLETED")
            metric_rows.append((run_id, execution_id, "reward", 0, 5.0))
    with sqlite3.connect(fixture["metrics_path"]) as connection:
        connection.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?)", metric_rows)

    dataset = load_experiment_dataset(fixture["db_path"], "dataset-test")

    assert dataset.runs["algo_version_id"].to_list() == [newer_algo, newer_algo]
    assert [item.code for item in dataset.diagnostics] == [DiagnosticCode.EXCLUDED_VERSION]
    assert {item.version_id for item in dataset.provenance} == {newer_algo, fixture["env_version"]}


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

    dataset = load_experiment_dataset(
        fixture["db_path"],
        "dataset-test",
        version_selection=VersionSelection.exact(
            (
                ComponentVersionRef.by_code_snapshot_hash("ppo", "ALGO", "algo-v1"),
                ComponentVersionRef.by_code_snapshot_hash("cartpole", "ENV", "env-v1"),
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
                (ComponentVersionRef.by_code_snapshot_hash("ppo", "ALGO", "algo-v-nonexistent"),)
            ),
        )

    assert error.value.diagnostics[0].code is DiagnosticCode.INVALID_VERSION_SELECTION


def test_missing_artifact_is_diagnosed_before_version_selection(tmp_path: Path) -> None:
    """A run with no recorded artifact is reported and cannot make its version set win selection."""
    db_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_id = database.add_experiment("broken-latest-test")
        algo_component = database.add_component("ppo", "ALGO")
        older_algo = database.add_component_version(algo_component, "algo-v1")
        newer_algo = database.add_component_version(algo_component, "algo-v2")
        env_version = database.add_component_version(database.add_component("cartpole", "ENV"), "env-v1")

        older_hyper_id = database.add_hyperparam_config({"condition_name": "baseline", "seed": 0})
        older_run_id = database.add_run(experiment_id, older_algo, env_version, older_hyper_id, 0)
        older_execution_id = database.add_execution()
        database.link_execution_run(older_execution_id, older_run_id)
        database.record_execution_artifacts(
            older_execution_id, str(tmp_path / "execution-older"), metadata={"metrics_db_path": str(metrics_path)}
        )
        database.update_execution_status(older_execution_id, "COMPLETED")

        newer_hyper_id = database.add_hyperparam_config({"condition_name": "broken", "seed": 1})
        newer_run_id = database.add_run(experiment_id, newer_algo, env_version, newer_hyper_id, 1)
        newer_execution_id = database.add_execution()
        database.link_execution_run(newer_execution_id, newer_run_id)
        database.update_execution_status(newer_execution_id, "COMPLETED")
        # Deliberately never record artifacts for the newer run.
    _write_metrics(metrics_path, [(older_run_id, older_execution_id, "reward", 0, 1.0)])

    dataset = load_experiment_dataset(db_path, "broken-latest-test", partial=True)

    assert dataset.runs["run_id"].to_list() == [older_run_id]
    assert [(item.code, item.run_id) for item in dataset.diagnostics] == [
        (DiagnosticCode.MALFORMED_ARTIFACT, newer_run_id)
    ]


def test_missing_recorded_metric_path_fails_without_root_fallback(tmp_path: Path) -> None:
    """Fail on an artifact without a recorded metric path even if root has a database."""
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

    assert [(item.code, item.run_id) for item in error.value.diagnostics] == [(DiagnosticCode.MALFORMED_ARTIFACT, run_id)]


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
