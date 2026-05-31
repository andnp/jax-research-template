"""Experiment-aware data loading for cross-project analysis workflows.

Bridges the two-database architecture (experiments DB + metrics DB) into a
single polars-first interface. The primary entry points are:

- :func:`resolve_run_artifacts` — resolve the latest completed execution per
  run and return a tidy metadata DataFrame.
- :func:`load_experiment_metrics` — load per-condition training metrics as a
  long-form polars DataFrame, ready for plotting and statistical analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from research_analysis.loader import load_sqlite_query

_ARTIFACTS_SCHEMA: dict[str, Any] = {
    "condition_name": pl.String,
    "seed": pl.Int64,
    "run_id": pl.Int64,
    "execution_id": pl.Int64,
    "metrics_db": pl.String,
}

_METRICS_SCHEMA: dict[str, Any] = {
    "condition_name": pl.String,
    "seed": pl.Int64,
    "step": pl.Int64,
    "metric": pl.String,
    "value": pl.Float64,
}


def resolve_run_artifacts(experiments_db: Path | str, slug: str) -> pl.DataFrame:
    """Resolve the latest completed execution artifacts for each run in an experiment.

    Returns a DataFrame with columns:

    - ``condition_name`` — experimental condition label
    - ``seed`` — random seed
    - ``run_id`` — database row id of the run
    - ``execution_id`` — database row id of the latest completed execution
    - ``metrics_db`` — path to the metrics SQLite database (may be ``null``
      if the execution artifacts predate metrics DB path recording)

    Only includes runs that have at least one completed execution with artifacts.

    Args:
        experiments_db: Path to the experiments SQLite database.
        slug: Experiment slug to resolve artifacts for.

    Raises:
        ValueError: If the experiment slug is not found in the database.
    """
    from experiment_definition.db import DatabaseManager

    rows: list[dict[str, object]] = []

    with DatabaseManager(Path(experiments_db)) as db:
        db.initialize()
        experiment = db.get_experiment(slug)
        if experiment is None:
            raise ValueError(f"Unknown experiment slug {slug!r} in {experiments_db}.")

        for run in db.list_runs(experiment.id):
            latest_exec = db.get_latest_completed_execution_for_run(run.id)
            latest_artifacts = db.get_latest_completed_artifacts_for_run(run.id)
            if latest_exec is None or latest_artifacts is None:
                continue

            hyper_config = db.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                continue

            hyperparameters: dict[str, object] = json.loads(hyper_config.json_blob)
            condition_name = str(hyperparameters.get("condition_name", ""))

            metadata: dict[str, object] = (
                json.loads(latest_artifacts.metadata_json) if latest_artifacts.metadata_json else {}
            )
            metrics_db_path = metadata.get("metrics_db_path")

            rows.append(
                {
                    "condition_name": condition_name,
                    "seed": run.seed,
                    "run_id": run.id,
                    "execution_id": latest_exec.id,
                    "metrics_db": metrics_db_path,
                }
            )

    return pl.DataFrame(rows or None, schema=_ARTIFACTS_SCHEMA)


def load_experiment_metrics(
    *,
    experiments_db: Path | str,
    slug: str,
    metrics: list[str] | None = None,
    run_artifacts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Load training metrics for all conditions in a completed experiment.

    Returns a tidy (long-form) DataFrame with columns:

    - ``condition_name`` — experimental condition label
    - ``seed`` — random seed
    - ``step`` — global training step
    - ``metric`` — metric name (e.g. ``"returned_episode_returns"``)
    - ``value`` — metric value at that step

    Only includes data from the most recently completed execution per run.
    Rows are sorted by ``(condition_name, seed, step, metric)``.

    Args:
        experiments_db: Path to the experiments SQLite database.
        slug: Experiment slug to load metrics for.
        metrics: Metric names to load. ``None`` loads all available metrics.
        run_artifacts: Optional pre-resolved artifacts DataFrame (from
            :func:`resolve_run_artifacts`). When omitted, resolved automatically
            from ``experiments_db`` and ``slug``.
    """
    if run_artifacts is None:
        run_artifacts = resolve_run_artifacts(experiments_db, slug)

    if run_artifacts.is_empty():
        return pl.DataFrame(schema=_METRICS_SCHEMA)

    metric_filter = ""
    if metrics is not None:
        if not metrics:
            return pl.DataFrame(schema=_METRICS_SCHEMA)
        quoted = ", ".join(f"'{m}'" for m in metrics)
        metric_filter = f" AND metric_name IN ({quoted})"

    frames: list[pl.DataFrame] = []
    for row in run_artifacts.iter_rows(named=True):
        if row["metrics_db"] is None:
            continue
        metrics_db_path = Path(str(row["metrics_db"]))
        if not metrics_db_path.exists():
            continue

        frame = load_sqlite_query(
            metrics_db_path,
            (
                "SELECT global_step, metric_name, value FROM metrics "
                f"WHERE run_id = {row['run_id']} AND execution_id = {row['execution_id']}"
                f"{metric_filter} ORDER BY metric_name, global_step"
            ),
        )

        if not frame.is_empty():
            frames.append(
                frame.rename({"global_step": "step", "metric_name": "metric"})
                .with_columns(
                    pl.lit(row["condition_name"]).alias("condition_name"),
                    pl.lit(row["seed"]).cast(pl.Int64).alias("seed"),
                )
                .select(list(_METRICS_SCHEMA.keys()))
            )

    if not frames:
        return pl.DataFrame(schema=_METRICS_SCHEMA)

    return pl.concat(frames).sort(["condition_name", "seed", "step", "metric"])
