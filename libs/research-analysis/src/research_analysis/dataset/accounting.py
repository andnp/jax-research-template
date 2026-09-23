"""Fold every planned run's outcome into per-cell accounting."""

from __future__ import annotations

from collections import Counter

import polars as pl

from research_analysis.dataset.candidates import RunState
from research_analysis.dataset.cohorts import cell_key
from research_analysis.dataset.rows import ResolvedRun
from research_analysis.dataset.types import CellAccounting, CellKey, ExclusionReason, RunAccounting
from research_analysis.dataset.versions import VersionSet
from research_analysis.diagnostics.types import DatasetDiagnostic, DiagnosticCode

_UNASSIGNED_REASONS = {
    DiagnosticCode.UNRESOLVED_DIMENSION: ExclusionReason.UNRESOLVED_DIMENSION,
    DiagnosticCode.MISSING_HYPER_CONFIG: ExclusionReason.MISSING_HYPER_CONFIG,
}


def _outcome(
    item: ResolvedRun,
    selected_sets: frozenset[VersionSet],
    missing_metrics: set[int | None],
    unreadable_metrics: set[int | None],
) -> RunState | ExclusionReason:
    if selected_sets and item.version.version_set not in selected_sets:
        return ExclusionReason.VERSION_NOT_SELECTED
    if item.candidate.state is not RunState.COMPLETED:
        return item.candidate.state
    if item.candidate.completed is None or item.candidate.run.id in unreadable_metrics:
        return ExclusionReason.MALFORMED_ARTIFACT
    if item.candidate.run.id in missing_metrics:
        return ExclusionReason.MISSING_METRIC_ARTIFACT
    return RunState.COMPLETED


def _cell_accounting(key: CellKey, outcomes: list[tuple[RunState | ExclusionReason, list[str]]]) -> CellAccounting:
    states = Counter(outcome for outcome, _metrics in outcomes)
    diverged = Counter(
        metric for outcome, metrics in outcomes if outcome is RunState.COMPLETED for metric in metrics
    )
    return CellAccounting(
        cell=key,
        planned=len(outcomes),
        completed=states[RunState.COMPLETED],
        failed=states[RunState.FAILED],
        pending=states[RunState.PENDING],
        diverged=dict(diverged),
        excluded={outcome: count for outcome, count in states.items() if isinstance(outcome, ExclusionReason)},
    )


def account_runs(
    resolved: list[ResolvedRun],
    selected_sets: frozenset[VersionSet],
    diagnostics: tuple[DatasetDiagnostic, ...],
    divergence: pl.DataFrame,
) -> RunAccounting:
    missing_metrics = {item.run_id for item in diagnostics if item.code is DiagnosticCode.MISSING_METRIC_ARTIFACT}
    unreadable_metrics = {item.run_id for item in diagnostics if item.code is DiagnosticCode.MALFORMED_ARTIFACT}
    diverged_metrics: dict[int, list[str]] = {}
    for run_id, metric in divergence.select(["run_id", "metric"]).iter_rows():
        diverged_metrics.setdefault(run_id, []).append(metric)

    outcomes: dict[CellKey, list[tuple[RunState | ExclusionReason, list[str]]]] = {}
    for item in resolved:
        run_id = item.candidate.run.id
        outcome = _outcome(item, selected_sets, missing_metrics, unreadable_metrics)
        outcomes.setdefault(cell_key(item), []).append((outcome, diverged_metrics.get(run_id, [])))

    unassigned = Counter(_UNASSIGNED_REASONS[item.code] for item in diagnostics if item.code in _UNASSIGNED_REASONS)
    return RunAccounting(
        cells=tuple(_cell_accounting(key, cell_outcomes) for key, cell_outcomes in outcomes.items()),
        unassigned=dict(unassigned),
    )
