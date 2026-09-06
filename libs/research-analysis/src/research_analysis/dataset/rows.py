"""Materialize typed rows for selected candidates, fetching each run's hyperparameters."""

from __future__ import annotations

from experiment_definition.db import DatabaseManager

from research_analysis.dataset.candidates import RunCandidate, VersionSet
from research_analysis.dataset.dimensions import build_run_row, cohort_matches
from research_analysis.dataset.types import CohortSelection, ExperimentDimensions, RunArtifacts
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


def build_rows(
    database: DatabaseManager,
    candidates: list[RunCandidate],
    selected_sets: frozenset[VersionSet],
    dimensions: ExperimentDimensions,
    cohort: CohortSelection | None,
    diagnostics: DiagnosticAccumulator,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], RunArtifacts]]:
    rows: list[dict[str, object]] = []
    artifacts: dict[tuple[int, int], RunArtifacts] = {}
    for candidate in candidates:
        if candidate.version_set not in selected_sets:
            continue
        hyper_config = database.get_hyperparam_config(candidate.run.hyper_id)
        if hyper_config is None:
            diagnostics.add(DiagnosticCode.MISSING_DIMENSION, "run has no hyperparameter configuration", candidate.run.id, candidate.execution.id)
            continue
        hyperparameters = hyper_config.hyperparameters
        if not isinstance(hyperparameters, dict):
            diagnostics.add(
                DiagnosticCode.MISSING_DIMENSION, "hyperparameter configuration is not an object", candidate.run.id, candidate.execution.id
            )
            continue
        row = build_run_row(
            candidate.run,
            candidate.execution,
            candidate.artifacts.metrics_db_path,
            candidate.provenance,
            hyperparameters,
            dimensions,
            diagnostics,
        )
        if row is None:
            continue
        if cohort is not None and not cohort_matches(row, cohort):
            continue
        rows.append(row)
        artifacts[(candidate.run.id, candidate.execution.id)] = candidate.artifacts
    return rows, artifacts
