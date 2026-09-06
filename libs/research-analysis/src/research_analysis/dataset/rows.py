"""Materialize typed rows for selected candidates, fetching each run's hyperparameters."""

from __future__ import annotations

from experiment_definition.db import DatabaseManager

from research_analysis.dataset.dimensions import cohort_matches, resolve_dimensions
from research_analysis.dataset.types import CohortSelection, ExperimentDimensions, RunArtifacts
from research_analysis.dataset.versions import VersionedCandidate, VersionSet


def build_rows(
    database: DatabaseManager,
    candidates: list[VersionedCandidate],
    selected_sets: frozenset[VersionSet],
    dimensions: ExperimentDimensions,
    cohort: CohortSelection | None,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], RunArtifacts]]:
    rows: list[dict[str, object]] = []
    artifacts: dict[tuple[int, int], RunArtifacts] = {}
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
        row: dict[str, object] = {
            "seed": completed_run.run.seed,
            "run_id": completed_run.run.id,
            "execution_id": completed_run.execution.id,
            "ablation": completed_run.run.ablation,
            "metrics_db_path": completed_run.artifacts.metrics_db_path,
            **dimension_values.values,
        }
        for item in candidate.provenance:
            prefix = "algo" if item.component_type == "ALGO" else "env"
            row[f"{prefix}_component_name"] = item.component_name
            row[f"{prefix}_component_type"] = item.component_type
            row[f"{prefix}_version_id"] = item.version_id
            row[f"{prefix}_version_number"] = item.version_number
            row[f"{prefix}_code_snapshot_hash"] = item.code_snapshot_hash
        if cohort is not None and not cohort_matches(row, cohort):
            continue
        rows.append(row)
        artifacts[(completed_run.run.id, completed_run.execution.id)] = completed_run.artifacts
    return rows, artifacts
