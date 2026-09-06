"""Materialize typed rows for selected candidates, fetching each run's hyperparameters."""

from __future__ import annotations

from experiment_definition.db import DatabaseManager

from research_analysis.dataset.candidates import RunCandidate
from research_analysis.dataset.dimensions import build_run_row, cohort_matches
from research_analysis.dataset.types import CohortSelection, ExperimentDimensions, RunArtifacts
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


def build_rows(
    database: DatabaseManager,
    candidates: list[RunCandidate],
    selected_sets: frozenset[tuple[int | None, int | None]],
    dimensions: ExperimentDimensions,
    cohort: CohortSelection | None,
    diagnostics: DiagnosticAccumulator,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], RunArtifacts]]:
    rows: list[dict[str, object]] = []
    artifacts: dict[tuple[int, int], RunArtifacts] = {}
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
        built = build_run_row(run, execution, artifact, items, hyperparameters, dimensions, diagnostics)
        if built is not None and (cohort is None or cohort_matches(built[0], cohort)):
            row, run_artifacts = built
            rows.append(row)
            artifacts[(run.id, execution.id)] = run_artifacts
    return rows, artifacts
