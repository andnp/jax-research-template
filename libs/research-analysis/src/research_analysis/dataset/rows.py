"""Materialize typed rows for selected candidates, fetching each run's hyperparameters."""

from __future__ import annotations

from experiment_definition.db import DatabaseManager

from research_analysis.dataset.dimensions import cohort_matches, resolve_dimensions
from research_analysis.dataset.types import CohortSelection, ExperimentDimensions, MaterializedRun
from research_analysis.dataset.versions import VersionedCandidate, VersionSet


def build_rows(
    database: DatabaseManager,
    candidates: list[VersionedCandidate],
    selected_sets: frozenset[VersionSet],
    dimensions: ExperimentDimensions,
    cohort: CohortSelection | None,
) -> list[MaterializedRun]:
    rows: list[MaterializedRun] = []
    for candidate in candidates:
        if candidate.version_set not in selected_sets:
            continue
        completed_run = candidate.completed_run
        hyper_config = database.get_hyperparam_config(completed_run.run.hyper_id)
        if hyper_config is None:
            continue

        dimension_values = resolve_dimensions(hyper_config.hyperparameters, dimensions)
        if dimension_values is None:
            continue

        row = MaterializedRun(
            dimension_values=dimension_values.values,
            seed=completed_run.run.seed,
            run_id=completed_run.run.id,
            execution_id=completed_run.execution.id,
            ablation=completed_run.run.ablation,
            provenance=candidate.provenance,
            artifacts=completed_run.artifacts.reference,
            metrics_db_path=completed_run.artifacts.metrics_db_path,
        )

        if cohort is not None and not cohort_matches(row, cohort):
            continue

        rows.append(row)
    return rows
