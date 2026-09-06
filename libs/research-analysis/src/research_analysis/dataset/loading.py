"""Canonical experiment dataset loading for analysis workflows."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl
from experiment_definition.db import DatabaseManager

from research_analysis.dataset.candidates import fetch_run_candidates
from research_analysis.dataset.dimensions import build_run_row, cohort_matches
from research_analysis.dataset.types import (
    CohortSelection,
    ExperimentDataset,
    ExperimentDimensions,
    RunArtifacts,
    SweepAxis,
    SweepValueType,
    VersionSelection,
)
from research_analysis.dataset.versions import select_version_sets
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import FATAL_DIAGNOSTIC_CODES, DatasetLoadError, DiagnosticCode
from research_analysis.loader import load_sqlite_query


def _axis_dtype(axis: SweepAxis):
    if axis.value_type is SweepValueType.STRING:
        return pl.String
    if axis.value_type is SweepValueType.INTEGER:
        return pl.Int64
    if axis.value_type is SweepValueType.FLOAT:
        return pl.Float64
    return pl.Boolean


def _required_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected integer dataset identity, got {value!r}")
    return value


def _metric_query_pairs(pairs: list[tuple[int, int]], metrics: tuple[str, ...] | None) -> str:
    pair_sql = " OR ".join(f"(run_id = {run_id} AND execution_id = {execution_id})" for run_id, execution_id in pairs)
    metric_sql = ""
    if metrics is not None:
        quoted = ", ".join("'" + metric.replace("'", "''") + "'" for metric in metrics)
        metric_sql = f" AND metric_name IN ({quoted})"
    return (
        "SELECT run_id, execution_id, global_step, metric_name, value FROM metrics "
        f"WHERE ({pair_sql}){metric_sql} ORDER BY run_id, execution_id, metric_name, global_step"
    )


def _load_metric_frames(
    rows: list[dict[str, object]], metrics: tuple[str, ...] | None, diagnostics: DiagnosticAccumulator
) -> list[pl.DataFrame]:
    metric_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    run_rows = {(_required_int(row["run_id"]), _required_int(row["execution_id"])): row for row in rows}
    for pair, row in run_rows.items():
        path = row["metrics_db_path"]
        if isinstance(path, str):
            metric_groups[path].append(pair)

    metric_frames: list[pl.DataFrame] = []
    for path, pairs in metric_groups.items():
        metric_path = Path(path)
        if not metric_path.exists():
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, f"metrics database does not exist: {metric_path}", run_id, execution_id)
            continue
        frame = load_sqlite_query(metric_path, _metric_query_pairs(pairs, metrics))
        if frame.is_empty():
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "metrics database has no requested rows", run_id, execution_id)
            continue
        metadata = pl.DataFrame([run_rows[pair] for pair in pairs])
        metric_frames.append(
            frame.join(metadata, on=["run_id", "execution_id"], how="inner")
            .rename({"global_step": "step", "metric_name": "metric"})
            .select(
                [
                    "condition_name",
                    *(["env_name"] if "env_name" in metadata.columns else []),
                    "seed",
                    "run_id",
                    "execution_id",
                    "ablation",
                    "step",
                    "metric",
                    "value",
                ]
            )
        )
    return metric_frames


def load_experiment_dataset(
    experiments_db: Path | str,
    slug: str,
    *,
    dimensions: ExperimentDimensions | None = None,
    version_selection: VersionSelection | None = None,
    cohort: CohortSelection | None = None,
    metrics: tuple[str, ...] | None = None,
    partial: bool = False,
) -> ExperimentDataset:
    """Load one typed experiment dataset from its recorded artifacts."""
    selected_dimensions = dimensions or ExperimentDimensions()
    selected_versions = version_selection or VersionSelection.latest_completed()
    diagnostics = DiagnosticAccumulator()
    rows: list[dict[str, object]] = []
    artifacts: dict[tuple[int, int], RunArtifacts] = {}

    with DatabaseManager(Path(experiments_db)) as database:
        database.initialize()
        experiment = database.get_experiment(slug)
        if experiment is None:
            raise ValueError(f"Unknown experiment slug {slug!r} in {experiments_db}.")

        candidates = fetch_run_candidates(database, experiment, diagnostics)
        selected_sets, all_provenance = select_version_sets(candidates, selected_versions, diagnostics)

        for run, execution, artifact, version_set, items in candidates:
            if version_set not in selected_sets:
                continue
            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                diagnostics.add(DiagnosticCode.MISSING_DIMENSION, "run has no hyperparameter configuration", run.id, execution.id)
                continue
            hyperparameters = hyper_config.hyperparameters
            if not isinstance(hyperparameters, dict):
                diagnostics.add(DiagnosticCode.MISSING_DIMENSION, "hyperparameter configuration is not an object", run.id, execution.id)
                continue
            built = build_run_row(run, execution, artifact, items, hyperparameters, selected_dimensions, diagnostics)
            if built is not None and (cohort is None or cohort_matches(built[0], cohort)):
                row, run_artifacts = built
                rows.append(row)
                artifacts[(run.id, execution.id)] = run_artifacts

    metric_frames = _load_metric_frames(rows, metrics, diagnostics)

    if rows:
        run_frame = pl.DataFrame(
            rows,
            schema_overrides={axis.name: _axis_dtype(axis) for axis in selected_dimensions.sweep_axes},
        )
    else:
        run_schema = {
            "condition_name": pl.String,
            **({"env_name": pl.String} if selected_dimensions.environment is not None else {}),
            "seed": pl.Int64,
            "run_id": pl.Int64,
            "execution_id": pl.Int64,
            "ablation": pl.String,
            "metrics_db_path": pl.String,
            **{axis.name: _axis_dtype(axis) for axis in selected_dimensions.sweep_axes},
        }
        run_frame = pl.DataFrame(schema=run_schema)
    metric_frame = pl.concat(metric_frames).sort(["condition_name", "seed", "step", "metric"]) if metric_frames else pl.DataFrame(
        schema={
            "condition_name": pl.String,
            **({"env_name": pl.String} if selected_dimensions.environment is not None else {}),
            "seed": pl.Int64,
            "run_id": pl.Int64,
            "execution_id": pl.Int64,
            "ablation": pl.String,
            "step": pl.Int64,
            "metric": pl.String,
            "value": pl.Float64,
        }
    )

    fatal = diagnostics.fatal(FATAL_DIAGNOSTIC_CODES)
    if fatal and not partial:
        raise DatasetLoadError(fatal)
    return ExperimentDataset(
        experiment_name=experiment.name,
        dimensions=selected_dimensions,
        runs=run_frame,
        metrics=metric_frame,
        provenance=tuple(all_provenance.values()),
        artifacts=artifacts,
        diagnostics=diagnostics.as_tuple(),
    )
