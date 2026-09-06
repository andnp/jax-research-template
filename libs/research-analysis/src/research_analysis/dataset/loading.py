"""Canonical experiment dataset loading for analysis workflows."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from experiment_definition.db import DatabaseManager

from research_analysis.dataset.candidates import fetch_run_candidates
from research_analysis.dataset.metrics import load_metric_frames
from research_analysis.dataset.rows import build_rows
from research_analysis.dataset.types import (
    CohortSelection,
    ExperimentDataset,
    ExperimentDimensions,
    SweepAxis,
    SweepValueType,
    VersionSelection,
)
from research_analysis.dataset.versions import select_version_sets
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import FATAL_DIAGNOSTIC_CODES, DatasetLoadError


def _axis_dtype(axis: SweepAxis):
    if axis.value_type is SweepValueType.STRING:
        return pl.String
    if axis.value_type is SweepValueType.INTEGER:
        return pl.Int64
    if axis.value_type is SweepValueType.FLOAT:
        return pl.Float64
    return pl.Boolean


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

    with DatabaseManager(Path(experiments_db)) as database:
        database.initialize()
        experiment = database.get_experiment(slug)
        if experiment is None:
            raise ValueError(f"Unknown experiment slug {slug!r} in {experiments_db}.")

        candidates = fetch_run_candidates(database, experiment, diagnostics)
        selected_sets, all_provenance = select_version_sets(candidates, selected_versions, diagnostics)
        rows, artifacts = build_rows(database, candidates, selected_sets, selected_dimensions, cohort, diagnostics)

    metric_frames = load_metric_frames(rows, metrics, diagnostics)

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
