from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from experiment_definition.db import DatabaseManager, ExperimentRow, PlannedExecution, RunRow
from experiment_definition.experiment import Experiment

from .git import capture_git_metadata
from .types import ExecutionContext, ExecutionResult


def run_experiment(
    db_path: Path,
    experiment: Experiment,
    train_fn: Callable[[ExecutionContext], ExecutionResult],
    *,
    executions_root: Path,
    metrics_db_path: Path | None = None,
    max_runs_per_batch: int | None = None,
    capture_git: bool = True,
    on_batch_complete: Callable[[Path], None] | None = None,
):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    experiment.sync(db_path)

    git_commit, git_diff = capture_git_metadata() if capture_git else (None, None)

    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(experiment.name)
        if experiment_row is None:
            raise RuntimeError(
                f"Experiment {experiment.name!r} was not synced into {db_path}.",
            )
        planned_batches = database.plan_experiment_execution_batches(
            experiment_row.id,
            executions_root,
            max_runs_per_batch=max_runs_per_batch,
            git_commit=git_commit,
            git_diff_blob=git_diff,
        )

    if not planned_batches:
        return []

    execution_roots: list[Path] = []
    for planned in planned_batches:
        root = execute_batch(
            db_path,
            planned,
            train_fn,
            metrics_db_path=metrics_db_path,
            capture_git=False,
        )
        execution_roots.append(root)
        if on_batch_complete is not None:
            on_batch_complete(root)

    return execution_roots


def execute_batch(
    db_path: Path,
    planned: PlannedExecution,
    train_fn: Callable[[ExecutionContext], ExecutionResult],
    *,
    metrics_db_path: Path | None = None,
    capture_git: bool = True,
):
    with DatabaseManager(db_path) as database:
        database.initialize()
        execution_root = Path(planned.root_path)
        manifest_path = Path(planned.manifest_path)
        runs = database.list_execution_runs(planned.execution_id)
        experiment_row = _resolve_experiment(database, runs)
        hyperparameters = _resolve_hyperparameters(database, runs)
        seed_values = tuple(run.seed for run in runs)
        resolved_metrics_db = metrics_db_path or _derive_metrics_db_path(execution_root)

        if capture_git:
            git_commit, git_diff = capture_git_metadata()
            database.record_execution_git_metadata(planned.execution_id, git_commit, git_diff)

        database.update_execution_status(
            planned.execution_id, "RUNNING", start_time=_utc_now(),
        )

    context = ExecutionContext(
        execution_id=planned.execution_id,
        experiment=experiment_row,
        runs=runs,
        hyperparameters=hyperparameters,
        seed_values=seed_values,
        execution_root=execution_root,
        metrics_db_path=resolved_metrics_db,
    )

    try:
        execution_root.mkdir(parents=True, exist_ok=True)
        result = train_fn(context)

        manifest = {
            "experiment_slug": experiment_row.name,
            "execution_id": planned.execution_id,
            "root_path": str(execution_root),
            "metrics_db_path": str(resolved_metrics_db),
            "seed_values": list(seed_values),
            "run_ids": [run.id for run in runs],
            "metadata": result.metadata,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with DatabaseManager(db_path) as database:
            database.initialize()
            database.record_execution_artifacts(
                planned.execution_id,
                str(execution_root),
                manifest_path=str(manifest_path),
                metadata={**result.metadata, "metrics_db_path": str(resolved_metrics_db)},
            )
            database.update_execution_status(
                planned.execution_id, "COMPLETED", end_time=_utc_now(),
            )
    except Exception:
        with DatabaseManager(db_path) as database:
            database.initialize()
            database.update_execution_status(
                planned.execution_id, "FAILED", end_time=_utc_now(),
            )
        raise

    return execution_root


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _resolve_experiment(database: DatabaseManager, runs: list[RunRow]):
    if not runs:
        raise ValueError("Execution has no linked logical runs.")
    row = database.conn.execute(
        "SELECT id, name, description, created_at FROM Experiments WHERE id = ?",
        (runs[0].experiment_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Experiment id {runs[0].experiment_id!r} is missing from the registry.",
        )
    return ExperimentRow(*row)


def _resolve_hyperparameters(database: DatabaseManager, runs: list[RunRow]):
    hyper_config = database.get_hyperparam_config(runs[0].hyper_id)
    if hyper_config is None:
        raise ValueError(f"Missing hyperparameter config for run {runs[0].id}.")
    return json.loads(hyper_config.json_blob)


def _derive_metrics_db_path(execution_root: Path):
    for candidate in (execution_root, *execution_root.parents):
        if candidate.name == "executions":
            return candidate.parent / "metrics.sqlite"
    raise ValueError(
        f"Could not derive metrics DB path from execution root {execution_root}. "
        "Pass metrics_db_path explicitly or use a conventional executions/ directory.",
    )
