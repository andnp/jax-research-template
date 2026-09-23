"""Select the winning component-version set under a VersionSelection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from experiment_definition.db import DatabaseManager
from experiment_definition.records import RunRow

from research_analysis.dataset.types import (
    CodeSnapshotSelector,
    ComponentVersionProvenance,
    ComponentVersionRef,
    VersionIdSelector,
    VersionSelection,
    VersionSelectionKind,
)
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


class VersionSet(NamedTuple):
    """Identifies one coherent algorithm/environment component-version pairing."""

    algo_version_id: int | None
    env_version_id: int | None


@dataclass(frozen=True)
class RunVersion:
    """One run's component-version pairing and the provenance of each component."""

    version_set: VersionSet
    provenance: tuple[ComponentVersionProvenance, ...]


def _component_provenance(database: DatabaseManager, version_id: int) -> ComponentVersionProvenance:
    row = database.get_component_version_provenance(version_id)
    if row is None:
        raise LookupError(f"Component version {version_id} does not exist")
    return ComponentVersionProvenance(
        component_id=row.component_id,
        component_name=row.component_name,
        component_type=row.component_type,
        version_id=row.version_id,
        version_number=row.version_number,
        code_snapshot_hash=row.code_snapshot_hash,
    )


def _cached_component_provenance(
    database: DatabaseManager, version_id: int, cache: dict[int, ComponentVersionProvenance]
) -> ComponentVersionProvenance:
    if version_id not in cache:
        cache[version_id] = _component_provenance(database, version_id)
    return cache[version_id]


def resolve_run_version(
    database: DatabaseManager, run: RunRow, cache: dict[int, ComponentVersionProvenance]
) -> RunVersion:
    provenance = tuple(
        _cached_component_provenance(database, version_id, cache)
        for version_id in (run.algo_version_id, run.env_version_id)
        if version_id is not None
    )
    return RunVersion(VersionSet(run.algo_version_id, run.env_version_id), provenance)


def _version_sort_key(version_set: VersionSet) -> tuple[int, int]:
    return version_set.algo_version_id or -1, version_set.env_version_id or -1


def _version_matches(
    provenance: tuple[ComponentVersionProvenance, ...], refs: tuple[ComponentVersionRef, ...]
) -> bool:
    for ref in refs:
        matches = [
            item
            for item in provenance
            if item.component_name == ref.component_name and item.component_type == ref.component_type
        ]
        if isinstance(ref.selector, VersionIdSelector):
            matches = [item for item in matches if item.version_id == ref.selector.value]
        elif isinstance(ref.selector, CodeSnapshotSelector):
            matches = [item for item in matches if item.code_snapshot_hash == ref.selector.value]
        if not matches:
            return False
    return True


def version_selected(version_set: VersionSet, selected_sets: frozenset[VersionSet], selection: VersionSelection) -> bool:
    """Whether a run's version set is selected; with no completed version to choose from, any set is."""
    if selected_sets or selection.kind is VersionSelectionKind.EXACT:
        return version_set in selected_sets
    return True


def select_version_sets(
    candidates: list[RunVersion], selection: VersionSelection, diagnostics: DiagnosticAccumulator
) -> tuple[frozenset[VersionSet], dict[int, ComponentVersionProvenance]]:
    available_sets = {candidate.version_set for candidate in candidates}
    selected_sets = available_sets
    if selection.kind is VersionSelectionKind.LATEST_COMPLETED and available_sets:
        selected_sets = {max(available_sets, key=_version_sort_key)}
    elif selection.kind is VersionSelectionKind.EXACT:
        selected_sets = {
            candidate.version_set for candidate in candidates if _version_matches(candidate.provenance, selection.exact_versions)
        }
        if not selected_sets:
            diagnostics.add(DiagnosticCode.INVALID_VERSION_SELECTION, "exact component versions have no completed runs")

    for version_set in sorted(available_sets, key=_version_sort_key):
        if version_set not in selected_sets:
            diagnostics.add(DiagnosticCode.EXCLUDED_VERSION, f"excluded completed version set {version_set!r}")

    all_provenance: dict[int, ComponentVersionProvenance] = {}
    for candidate in candidates:
        if candidate.version_set in selected_sets:
            all_provenance.update({item.version_id: item for item in candidate.provenance})

    return frozenset(selected_sets), all_provenance
