"""Detect cohorts whose cells the requested selection cannot identify unambiguously."""

from __future__ import annotations

from research_analysis.dataset.rows import ResolvedRun
from research_analysis.dataset.types import SweepValue, VersionSelection, VersionSelectionKind
from research_analysis.dataset.versions import VersionSet
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode

CellKey = tuple[tuple[str, SweepValue | None], ...]


def cell_key(item: ResolvedRun) -> CellKey:
    return (*item.dimension_values.items(), ("ablation", item.candidate.run.ablation))


def diagnose_orphaned_cells(
    resolved: list[ResolvedRun],
    selection: VersionSelection,
    selected_sets: frozenset[VersionSet],
    diagnostics: DiagnosticAccumulator,
) -> None:
    if selection.kind is not VersionSelectionKind.LATEST_COMPLETED:
        return
    completed = [item for item in resolved if item.candidate.completed is not None]
    covered = {cell_key(item) for item in completed if item.version.version_set in selected_sets}
    orphaned: dict[CellKey, set[VersionSet]] = {}
    for item in completed:
        key = cell_key(item)
        if key not in covered:
            orphaned.setdefault(key, set()).add(item.version.version_set)

    for key, version_sets in orphaned.items():
        diagnostics.add(
            DiagnosticCode.AMBIGUOUS_COHORT,
            f"latest_completed selected {sorted(selected_sets)} but cell {dict(key)} has completed runs only in "
            f"{sorted(version_sets)}; finish re-running it under the selected versions or choose "
            "VersionSelection.exact(...) or VersionSelection.all() explicitly",
        )


def _varying_keys(runs: list[ResolvedRun]) -> list[str]:
    keys = {key for item in runs for key in item.hyperparameters} - {"seed"}
    return sorted(
        key
        for key in keys
        if len({(key in item.hyperparameters, item.hyperparameters.get(key)) for item in runs}) > 1
    )


def diagnose_undeclared_variation(
    resolved: list[ResolvedRun], selected_sets: frozenset[VersionSet], diagnostics: DiagnosticAccumulator
) -> None:
    cells: dict[CellKey, list[ResolvedRun]] = {}
    for item in resolved:
        if item.candidate.completed is not None and item.version.version_set in selected_sets:
            cells.setdefault(cell_key(item), []).append(item)

    for key, runs in cells.items():
        varying = _varying_keys(runs)
        if varying:
            diagnostics.add(
                DiagnosticCode.AMBIGUOUS_COHORT,
                f"cell {dict(key)} pools runs whose undeclared hyperparameters {varying} differ; declare them as "
                "SweepAxis entries (or as the condition/environment dimension) so each configuration is its own cell",
            )
