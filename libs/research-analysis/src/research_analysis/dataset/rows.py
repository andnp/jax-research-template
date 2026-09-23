"""Resolve every planned run's dimensions, then materialize the selected completed rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from experiment_definition.db import DatabaseManager
from experiment_definition.parameter import ParameterValue

from research_analysis.dataset.candidates import RunCandidate, RunState
from research_analysis.dataset.dimensions import UnresolvedDimension, cohort_matches, resolve_dimensions
from research_analysis.dataset.types import (
    CohortSelection,
    ComponentVersionProvenance,
    ExperimentDimensions,
    MaterializedRun,
    SweepValue,
)
from research_analysis.dataset.versions import RunVersion, VersionSet, resolve_run_version
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


@dataclass(frozen=True)
class ResolvedRun:
    """A planned run with its analysis dimensions, hyperparameters, and component versions."""

    candidate: RunCandidate
    version: RunVersion
    dimension_values: Mapping[str, SweepValue | None]
    hyperparameters: Mapping[str, ParameterValue]


def resolve_runs(
    database: DatabaseManager,
    candidates: list[RunCandidate],
    dimensions: ExperimentDimensions,
    cohort: CohortSelection | None,
    diagnostics: DiagnosticAccumulator,
) -> list[ResolvedRun]:
    provenance_cache: dict[int, ComponentVersionProvenance] = {}
    resolved: list[ResolvedRun] = []
    for candidate in candidates:
        run = candidate.run
        hyper_config = database.get_hyperparam_config(run.hyper_id)
        if hyper_config is None:
            diagnostics.add(
                DiagnosticCode.MISSING_HYPER_CONFIG, f"run {run.id} references missing hyperparameter config {run.hyper_id}", run.id
            )
            continue

        dimension_values = resolve_dimensions(hyper_config.hyperparameters, dimensions)
        if isinstance(dimension_values, UnresolvedDimension):
            diagnostics.add(
                DiagnosticCode.UNRESOLVED_DIMENSION, f"run {run.id} (seed {run.seed}): {dimension_values.reason}", run.id
            )
            continue

        if cohort is not None and not cohort_matches(dimension_values.values, run.ablation, cohort):
            continue

        resolved.append(
            ResolvedRun(
                candidate=candidate,
                version=resolve_run_version(database, run, provenance_cache),
                dimension_values=dimension_values.values,
                hyperparameters=hyper_config.hyperparameters,
            )
        )
    return resolved


def materialize_rows(resolved: list[ResolvedRun], selected_sets: frozenset[VersionSet]) -> list[MaterializedRun]:
    rows: list[MaterializedRun] = []
    for item in resolved:
        completed = item.candidate.completed
        if completed is None or item.version.version_set not in selected_sets:
            continue
        rows.append(
            MaterializedRun(
                dimension_values=item.dimension_values,
                seed=completed.run.seed,
                run_id=completed.run.id,
                execution_id=completed.execution.id,
                ablation=completed.run.ablation,
                provenance=item.version.provenance,
                artifacts=completed.artifacts.reference,
                metrics_db_path=completed.artifacts.metrics_db_path,
            )
        )
    return rows


def diagnose_unusable_runs(
    resolved: list[ResolvedRun], selected_sets: frozenset[VersionSet], diagnostics: DiagnosticAccumulator
) -> None:
    for item in resolved:
        run = item.candidate.run
        if item.candidate.state is RunState.COMPLETED:
            if item.candidate.completed is None:
                diagnostics.add(
                    DiagnosticCode.MALFORMED_ARTIFACT,
                    f"run {run.id} (seed {run.seed}) completed without a valid artifact record naming its metrics database",
                    run.id,
                )
            continue
        if selected_sets and item.version.version_set not in selected_sets:
            continue
        if item.candidate.state is RunState.FAILED:
            diagnostics.add(DiagnosticCode.RUN_FAILED, f"run {run.id} (seed {run.seed}) has only failed executions", run.id)
        else:
            diagnostics.add(DiagnosticCode.RUN_PENDING, f"run {run.id} (seed {run.seed}) has not completed yet", run.id)
