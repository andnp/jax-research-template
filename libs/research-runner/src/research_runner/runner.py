from __future__ import annotations

import json
import os
import resource
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from experiment_definition.db import DatabaseManager, PlannedExecution
from experiment_definition.parameter import ParameterValue
from experiment_definition.records import RunRow
from experiment_definition.status import ExecutionStatus
from research_instrument.scalar import write_execution_scalars, write_seed_batch_scalars
from research_instrument.seed_batch import write_seed_batch_curve
from research_store.store import Store

from .git import capture_git_metadata
from .types import ExecutionContext, ExecutionResult, ExperimentSpec, RunPoint


def run_experiment(
    spec: ExperimentSpec,
    *,
    max_runs_per_batch: int | None = None,
    on_batch_complete: Callable[[Path], None] | None = None,
    reclaim_stale_after_seconds: float | None = None,
    only: Mapping[str, str] | None = None,
    group: str | None = None,
):
    """Sync the spec, then plan and execute its pending batches until none remain.

    ``max_runs_per_batch`` overrides the spec's. ``only`` restricts which pending runs
    execute, per the planning service matcher. ``group`` restricts execution to the
    batches whose ``RunBatch.group_key`` matches.
    """
    db_path = spec.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    spec.sync(db_path)

    git_commit, git_diff = capture_git_metadata() if spec.capture_git else (None, None)

    with DatabaseManager(db_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(spec.experiment.name)
        if experiment_row is None:
            raise RuntimeError(
                f"Experiment {spec.experiment.name!r} was not synced into {db_path}.",
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
                spec.executions_root,
                max_runs_per_batch=max_runs_per_batch if max_runs_per_batch is not None else spec.max_runs_per_batch,
                git_commit=git_commit,
                git_diff_blob=git_diff,
                only=only,
                group=group,
            )
        if planned is None:
            break

        root = execute_batch(
            db_path,
            planned,
            spec,
            capture_git=False,
        )
        execution_roots.append(root)
        if on_batch_complete is not None:
            on_batch_complete(root)

    return execution_roots


def execute_batch(
    db_path: Path,
    planned: PlannedExecution,
    spec: ExperimentSpec,
    *,
    capture_git: bool = True,
):
    try:
        with DatabaseManager(db_path) as database:
            database.initialize()
            execution_root = Path(planned.root_path)
            manifest_path = Path(planned.manifest_path)
            runs = database.list_execution_runs(planned.execution_id)
            experiment_row = _resolve_experiment(database, runs)
            static_config = planned.static_config
            points = _resolve_points(database, runs, static_config)
            resolved_metrics_db = spec.metrics_db_path

            if capture_git:
                git_commit, git_diff = capture_git_metadata()
                database.record_execution_git_metadata(
                    planned.execution_id, git_commit, git_diff,
                )

            database.update_execution_status(
                planned.execution_id, ExecutionStatus.RUNNING, start_time=_utc_now(),
            )
    except Exception:
        with DatabaseManager(db_path) as database:
            database.initialize()
            database.fail_execution(planned.execution_id)
        raise

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
        started = time.perf_counter()
        result = spec.train_fn(context)
        _write_metrics(context, result, spec.metric_subsample_factor)
        artifact_uris = _write_artifacts(context, result)
        metadata: dict[str, object] = {"runtime": _runtime_telemetry(len(points), time.perf_counter() - started)}
        if artifact_uris:
            metadata["artifacts"] = artifact_uris

        manifest = {
            "experiment_slug": experiment_row.name,
            "execution_id": planned.execution_id,
            "root_path": str(execution_root),
            "metrics_db_path": str(resolved_metrics_db),
            "run_ids": [point.run_id for point in points],
            "metadata": metadata,
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
                metadata={**metadata, "metrics_db_path": str(resolved_metrics_db)},
            )
            database.update_execution_status(
                planned.execution_id, ExecutionStatus.COMPLETED, end_time=_utc_now(),
            )
    except Exception:
        with DatabaseManager(db_path) as database:
            database.initialize()
            database.fail_execution(planned.execution_id)
        raise

    return execution_root


def _write_artifacts(context: ExecutionContext, result: ExecutionResult) -> dict[str, str]:
    if not result.artifacts:
        return {}
    execution_id = uuid5(NAMESPACE_URL, f"{context.experiment_name}/{context.execution_id}")
    store = Store(root=context.execution_root / "artifacts", experiment_id=context.experiment_name)
    uris = {
        name: str(store.put(blob, name=name, execution_id=execution_id))
        for name, blob in result.artifacts.items()
    }
    store.sync()
    return uris


def _write_metrics(
    context: ExecutionContext,
    result: ExecutionResult,
    subsample_factor: int,
) -> None:
    run_ids = tuple(point.run_id for point in context.points)
    seed_ids: list[int] = []
    for point in context.points:
        seed = point.hyperparameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"Run {point.run_id} seed must be an int, got {seed!r}.")
        seed_ids.append(seed)

    for metric_name, curves in result.metric_curves.items():
        write_seed_batch_curve(
            context.metrics_db_path,
            [row[::subsample_factor] for row in curves],
            metric_name=metric_name,
            experiment_id=context.experiment_id,
            execution_id=context.execution_id,
            run_ids=run_ids,
            seed_ids=tuple(seed_ids),
            every=subsample_factor,
        )

    write_seed_batch_scalars(
        context.metrics_db_path,
        result.run_scalars,
        experiment_id=context.experiment_id,
        execution_id=context.execution_id,
        run_ids=run_ids,
        seed_ids=tuple(seed_ids),
    )

    write_execution_scalars(
        context.metrics_db_path,
        result.execution_scalars,
        experiment_id=context.experiment_id,
        execution_id=context.execution_id,
    )


def _runtime_telemetry(batch_runs: int, wall_seconds: float) -> dict[str, int | float | None]:
    nproc = os.environ.get("NPROC")
    return {
        "nproc": int(nproc) if nproc is not None else None,
        "batch_runs": batch_runs,
        "wall_seconds": wall_seconds,
        # ru_maxrss is in KiB on Linux
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


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
    static_config: Mapping[str, ParameterValue],
) -> tuple[RunPoint, ...]:
    config_cache: dict[int, dict[str, ParameterValue]] = {}
    points: list[RunPoint] = []
    for run in runs:
        if run.hyper_id not in config_cache:
            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                raise ValueError(f"Missing hyperparameter config for run {run.id}.")
            config_cache[run.hyper_id] = dict(hyper_config.hyperparameters)
        hyperparameters = {**config_cache[run.hyper_id], "seed": run.seed}
        for key, value in static_config.items():
            if hyperparameters.get(key) != value:
                raise ValueError(
                    f"Run {run.id} static parameter {key!r} ({hyperparameters.get(key)!r}) "
                    f"does not match the batch's static config value {value!r}.",
                )
        points.append(RunPoint(run_id=run.id, hyperparameters=hyperparameters))
    return tuple(points)
