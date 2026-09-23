"""Canonical experiment dataset loading for analysis workflows."""

from __future__ import annotations

from pathlib import Path

from experiment_definition.db import DatabaseManager

from research_analysis.dataset.candidates import fetch_run_candidates
from research_analysis.dataset.cohorts import diagnose_orphaned_cells, diagnose_undeclared_variation
from research_analysis.dataset.frames import assemble_frames
from research_analysis.dataset.metrics import load_metric_frames
from research_analysis.dataset.non_finite import diagnose_non_finite, non_finite_summary, with_non_finite_metrics
from research_analysis.dataset.rows import diagnose_unusable_runs, materialize_rows, resolve_runs
from research_analysis.dataset.types import CohortSelection, ExperimentDataset, ExperimentDimensions, VersionSelection
from research_analysis.dataset.versions import select_version_sets
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import FATAL_DIAGNOSTIC_CODES, DatasetLoadError


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

        run_candidates = fetch_run_candidates(database, experiment)
        resolved = resolve_runs(database, run_candidates, selected_dimensions, cohort, diagnostics)

    completed_versions = [item.version for item in resolved if item.candidate.completed is not None]
    selected_sets, all_provenance = select_version_sets(completed_versions, selected_versions, diagnostics)
    diagnose_orphaned_cells(resolved, selected_versions, selected_sets, diagnostics)
    diagnose_undeclared_variation(resolved, selected_sets, diagnostics)
    diagnose_unusable_runs(resolved, selected_sets, diagnostics)
    rows = materialize_rows(resolved, selected_sets)

    metric_frames = load_metric_frames(rows, metrics, diagnostics)
    run_frame, metric_frame = assemble_frames(rows, metric_frames, selected_dimensions)
    divergence = non_finite_summary(metric_frame)
    diagnose_non_finite(divergence, diagnostics)
    run_frame = with_non_finite_metrics(run_frame, divergence)
    artifacts = {(row.run_id, row.execution_id): row.artifacts for row in rows}

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
