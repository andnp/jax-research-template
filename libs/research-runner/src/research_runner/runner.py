from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from experiment_definition import Experiment, ParameterValue
from experiment_definition.db import DatabaseManager, PlannedExecution, RunRow

from .git import capture_git_metadata
from .types import ExecutionContext, ExecutionResult, RunPoint


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
    reclaim_stale_after_seconds: float | None = None,
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
        experiment_id = experiment_row.id

    execution_roots: list[Path] = []
    while True:
        with DatabaseManager(db_path) as database:
            database.initialize()
            if reclaim_stale_after_seconds is not None:
                database.reclaim_stale_executions(experiment_id, reclaim_stale_after_seconds)
            planned = database.plan_next_execution_batch(
                experiment_id,
                executions_root,
                max_runs_per_batch=max_runs_per_batch,
                git_commit=git_commit,
                git_diff_blob=git_diff,
            )
        if planned is None:
            break

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
        static_config = json.loads(planned.static_config_json)
        points = _resolve_points(database, runs, static_config)
        resolved_metrics_db = metrics_db_path or db_path.parent / "metrics.sqlite"

        if capture_git:
            git_commit, git_diff = capture_git_metadata()
            database.record_execution_git_metadata(planned.execution_id, git_commit, git_diff)

        database.update_execution_status(
            planned.execution_id, "RUNNING", start_time=_utc_now(),
        )

    context = ExecutionContext(
        execution_id=planned.execution_id,
        experiment_id=experiment_row.id,
        experiment_name=experiment_row.name,
        static_config=static_config,
        points=points,
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
            "run_ids": [point.run_id for point in points],
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
    experiment_row = database.get_experiment_by_id(runs[0].experiment_id)
    if experiment_row is None:
        raise RuntimeError(
            f"Experiment id {runs[0].experiment_id!r} is missing from the registry.",
        )
    return experiment_row


def _resolve_points(
    database: DatabaseManager,
    runs: list[RunRow],
    static_config: dict[str, ParameterValue],
) -> tuple[RunPoint, ...]:
    config_cache: dict[int, dict[str, ParameterValue]] = {}
    points: list[RunPoint] = []
    for run in runs:
        if run.hyper_id not in config_cache:
            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                raise ValueError(f"Missing hyperparameter config for run {run.id}.")
            config_cache[run.hyper_id] = json.loads(hyper_config.json_blob)
        hyperparameters = {**config_cache[run.hyper_id], "seed": run.seed}
        for key, value in static_config.items():
            if hyperparameters.get(key) != value:
                raise ValueError(
                    f"Run {run.id} static parameter {key!r} ({hyperparameters.get(key)!r}) "
                    f"does not match the batch's static config value {value!r}.",
                )
        points.append(RunPoint(run_id=run.id, hyperparameters=hyperparameters))
    return tuple(points)
