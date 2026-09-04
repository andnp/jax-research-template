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
from research_analysis.schema import AnalysisSchema

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

_MAPPED_ARTIFACTS_SCHEMA: dict[str, Any] = {
    "condition_name": pl.String,
    "env_name": pl.String,
    "seed": pl.Int64,
    "run_id": pl.Int64,
    "execution_id": pl.Int64,
    "metrics_db": pl.String,
}

_MAPPED_METRICS_SCHEMA: dict[str, Any] = {
    "condition_name": pl.String,
    "env_name": pl.String,
    "seed": pl.Int64,
    "step": pl.Int64,
    "metric": pl.String,
    "value": pl.Float64,
}


def _max_component_version(values: list[int | None]) -> int | None:
    """Return the maximum of the non-``None`` values, or ``None`` if all are ``None``."""
    present = [v for v in values if v is not None]
    return max(present) if present else None


def resolve_run_artifacts(
    experiments_db: Path | str,
    slug: str,
    *,
    schema: AnalysisSchema | None = None,
    latest_component_versions: bool = True,
) -> pl.DataFrame:
    """Resolve the latest completed execution artifacts for each run in an experiment.

    Returns a DataFrame with columns:

    - ``condition_name`` — experimental condition label
    - ``seed`` — random seed
    - ``run_id`` — database row id of the run
    - ``execution_id`` — database row id of the latest completed execution
    - ``metrics_db`` — path to the metrics SQLite database (may be ``null``
      if the execution artifacts predate metrics DB path recording)

    Only includes runs that have at least one completed execution with artifacts.

    By default (``latest_component_versions=True``) only runs whose
    ``algo_version_id`` and ``env_version_id`` both equal the maximum values
    among runs *with completed executions* are returned — i.e. the newest
    version that actually has results, not merely the newest version that
    exists. ``sync`` mints a new ComponentVersion and inserts unexecuted Run
    rows for it the moment a component's source hash changes, including for
    a behaviour-preserving refactor; if the max were taken over all runs,
    the newest version would routinely have zero completed runs and this
    function would silently return an empty frame for a study that is in
    fact complete at the prior version. Without this restriction at all, an
    experiment re-run after an agent or environment fix silently pools runs
    from every prior code version into one result, averaging broken and fixed
    behaviour together with no error or warning. ``algo_version_id`` is
    ``None`` whenever the experiment declares more than one ALGO component;
    ``None`` is matched against ``None`` (an experiment with no versioned
    runs still returns them), never coerced to "oldest" or dropped. Pass
    ``False`` to restore the old pooling behaviour, e.g. to deliberately
    compare a pre-fix arm against a post-fix arm.

    Args:
        experiments_db: Path to the experiments SQLite database.
        slug: Experiment slug to resolve artifacts for.

    Raises:
        ValueError: If the experiment slug is not found in the database.
    """
    from experiment_definition.db import DatabaseManager, ExecutionArtifactRow, ExecutionRow, RunRow

    rows: list[dict[str, object]] = []

    with DatabaseManager(Path(experiments_db)) as db:
        db.initialize()
        experiment = db.get_experiment(slug)
        if experiment is None:
            raise ValueError(f"Unknown experiment slug {slug!r} in {experiments_db}.")

        completed: list[tuple[RunRow, ExecutionRow, ExecutionArtifactRow]] = []
        for run in db.list_runs(experiment.id):
            latest_exec = db.get_latest_completed_execution_for_run(run.id)
            latest_artifacts = db.get_latest_completed_artifacts_for_run(run.id)
            if latest_exec is None or latest_artifacts is None:
                continue
            completed.append((run, latest_exec, latest_artifacts))

        if latest_component_versions and completed:
            max_algo = _max_component_version([r.algo_version_id for r, _, _ in completed])
            max_env = _max_component_version([r.env_version_id for r, _, _ in completed])
            completed = [
                (r, e, a)
                for r, e, a in completed
                if r.algo_version_id == max_algo and r.env_version_id == max_env
            ]

        for run, latest_exec, latest_artifacts in completed:
            hyper_config = db.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                continue

            hyperparameters: dict[str, object] = json.loads(hyper_config.json_blob)
            condition_name = (
                schema.condition_name(hyperparameters)
                if schema is not None
                else str(hyperparameters.get("condition_name", ""))
            )

            metadata: dict[str, object] = (
                json.loads(latest_artifacts.metadata_json) if latest_artifacts.metadata_json else {}
            )
            metrics_db_path = metadata.get("metrics_db_path")

            row: dict[str, object] = {
                "condition_name": condition_name,
                "seed": run.seed,
                "run_id": run.id,
                "execution_id": latest_exec.id,
                "metrics_db": metrics_db_path,
            }
            if schema is not None:
                row["env_name"] = schema.environment_name(hyperparameters)
            rows.append(row)

    result_schema = _MAPPED_ARTIFACTS_SCHEMA if schema is not None else _ARTIFACTS_SCHEMA
    return pl.DataFrame(rows or None, schema=result_schema)


def load_experiment_metrics(
    *,
    experiments_db: Path | str,
    slug: str,
    metrics: list[str] | None = None,
    run_artifacts: pl.DataFrame | None = None,
    schema: AnalysisSchema | None = None,
    latest_component_versions: bool = True,
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

    ``latest_component_versions`` is passed through to
    :func:`resolve_run_artifacts` when ``run_artifacts`` is not supplied
    directly; see that function's docstring for its default-restrictive
    behaviour. It has no effect when ``run_artifacts`` is provided — the
    caller has already made that choice.

    Args:
        experiments_db: Path to the experiments SQLite database.
        slug: Experiment slug to load metrics for.
        metrics: Metric names to load. ``None`` loads all available metrics.
        run_artifacts: Optional pre-resolved artifacts DataFrame (from
            :func:`resolve_run_artifacts`). When omitted, resolved automatically
            from ``experiments_db`` and ``slug``.
    """
    if run_artifacts is None:
        run_artifacts = resolve_run_artifacts(
            experiments_db,
            slug,
            schema=schema,
            latest_component_versions=latest_component_versions,
        )

    mapped = "env_name" in run_artifacts.columns
    if run_artifacts.is_empty():
        result_schema = _MAPPED_METRICS_SCHEMA if mapped else _METRICS_SCHEMA
        return pl.DataFrame(schema=result_schema)

    metric_filter = ""
    if metrics is not None:
        if not metrics:
            result_schema = _MAPPED_METRICS_SCHEMA if mapped else _METRICS_SCHEMA
            return pl.DataFrame(schema=result_schema)
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
            )
            if mapped:
                frames[-1] = frames[-1].with_columns(
                    pl.lit(row["env_name"]).alias("env_name")
                )
            result_columns = _MAPPED_METRICS_SCHEMA if mapped else _METRICS_SCHEMA
            frames[-1] = frames[-1].select(list(result_columns.keys()))

    if not frames:
        result_schema = _MAPPED_METRICS_SCHEMA if mapped else _METRICS_SCHEMA
        return pl.DataFrame(schema=result_schema)

    sort_columns = ["condition_name", "seed", "step", "metric"]
    if mapped:
        sort_columns.insert(1, "env_name")
    return pl.concat(frames).sort(sort_columns)
