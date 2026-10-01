from __future__ import annotations

import json
import os
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from experiment_definition.db import DatabaseManager, PlannedExecution
from experiment_definition.metric import MetricSpec
from experiment_definition.ownership import local_owner
from experiment_definition.parameter import ParameterValue
from experiment_definition.records import RunRow
from experiment_definition.status import ExecutionStatus
from research_instrument.scalar import write_execution_scalars, write_seed_batch_scalars
from research_instrument.seed_batch import write_seed_batch_curve

from .batch import _run_batch
from .config import _build_config
from .git import capture_git_metadata
from .types import ExperimentSpec


@dataclass(frozen=True, slots=True)
class _RunPoint:
    """One logical run's resolved hyperparameters, including its seed."""

    run_id: int
    hyperparameters: Mapping[str, ParameterValue]


def run_experiment(
    spec: ExperimentSpec,
    *,
    max_runs_per_batch: int | None = None,
    on_batch_complete: Callable[[Path], None] | None = None,
    only: Mapping[str, str] | None = None,
):
    """Sync the spec, then plan and execute its pending batches until none remain.

    ``max_runs_per_batch`` overrides the spec's. ``only`` restricts which pending runs
    execute, per the planning service matcher.
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

    owner = local_owner()
    execution_roots: list[Path] = []
    while True:
        with DatabaseManager(db_path) as database:
            database.initialize()
            planned = database.plan_next_execution_batch(
                experiment_id,
                spec.executions_root,
                max_runs_per_batch=max_runs_per_batch if max_runs_per_batch is not None else spec.max_runs_per_batch,
                git_commit=git_commit,
                git_diff_blob=git_diff,
                only=only,
                owner=owner,
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
    with DatabaseManager(db_path) as database:
        database.initialize()
        runs = database.list_execution_runs(planned.execution_id)
        experiment_row = _resolve_experiment(database, runs)
    if experiment_row.name != spec.experiment.name:
        raise ValueError(
            f"Execution {planned.execution_id} belongs to experiment {experiment_row.name!r}, "
            f"not to the spec's experiment {spec.experiment.name!r}.",
        )

    try:
        with DatabaseManager(db_path) as database:
            database.initialize()
            execution_root = Path(planned.root_path)
            manifest_path = Path(planned.manifest_path)
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

    try:
        execution_root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        configs = [_build_config(spec.config, spec.choices, point.hyperparameters) for point in points]
        seeds = [run.seed for run in runs]
        metrics = _metrics_for(spec.experiment.metrics, points)
        outputs = _run_batch(spec.train, metrics, configs, seeds)
        _write_metrics(
            resolved_metrics_db,
            outputs,
            metrics,
            experiment_id=experiment_row.id,
            execution_id=planned.execution_id,
            run_ids=[point.run_id for point in points],
            seeds=seeds,
        )
        metadata: dict[str, object] = {
            "runtime": _runtime_telemetry(len(points), time.perf_counter() - started),
            "metrics": {metric.name: {"every": metric.every, "ema": metric.ema} for metric in metrics},
        }

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


def _metrics_for(metrics: Sequence[MetricSpec], points: Sequence[_RunPoint]) -> list[MetricSpec]:
    """Keep the declarations whose ``when()`` conditions every run in the batch satisfies."""
    selected: list[MetricSpec] = []
    for metric in metrics:
        for key in metric.conditions:
            if all(key not in point.hyperparameters for point in points):
                raise ValueError(f"metric {metric.name!r} is conditioned on {key!r}, which no run in this batch declares")
        matches = [
            all(point.hyperparameters.get(key) == value for key, value in metric.conditions.items()) for point in points
        ]
        if all(matches):
            selected.append(metric)
        elif any(matches):
            raise ValueError(
                f"metric {metric.name!r} with conditions {metric.conditions} matches only some runs in this batch; "
                "conditional metrics must be conditioned on static parameters",
            )
    return selected


def _write_metrics(
    metrics_db_path: Path,
    outputs: Mapping[str, np.ndarray],
    metrics: Sequence[MetricSpec],
    *,
    experiment_id: int,
    execution_id: int,
    run_ids: Sequence[int],
    seeds: Sequence[int],
) -> None:
    """Record each output by its shape: ``[runs, T]`` is a curve, ``[runs]`` a per-run scalar, ``()`` a batch scalar."""
    every = {metric.name: metric.every for metric in metrics}
    run_scalars: dict[str, list[float]] = {}
    execution_scalars: dict[str, float] = {}
    for name, values in outputs.items():
        if values.ndim == 2:
            write_seed_batch_curve(
                metrics_db_path,
                values.tolist(),
                metric_name=name,
                experiment_id=experiment_id,
                execution_id=execution_id,
                run_ids=run_ids,
                seed_ids=seeds,
                every=every[name],
            )
        elif values.ndim == 1:
            run_scalars[name] = values.tolist()
        else:
            execution_scalars[name] = float(values)
    write_seed_batch_scalars(
        metrics_db_path,
        run_scalars,
        experiment_id=experiment_id,
        execution_id=execution_id,
        run_ids=run_ids,
        seed_ids=seeds,
    )
    write_execution_scalars(metrics_db_path, execution_scalars, experiment_id=experiment_id, execution_id=execution_id)


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
) -> tuple[_RunPoint, ...]:
    config_cache: dict[int, dict[str, ParameterValue]] = {}
    points: list[_RunPoint] = []
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
        points.append(_RunPoint(run_id=run.id, hyperparameters=hyperparameters))
    return tuple(points)
