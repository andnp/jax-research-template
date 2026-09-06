"""Select the winning component-version set under a VersionSelection policy."""

from __future__ import annotations

from research_analysis.dataset.candidates import RunCandidate, VersionSet
from research_analysis.dataset.types import ComponentVersionProvenance, ComponentVersionRef, VersionSelection, VersionSelectionKind
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


def _version_sort_key(version_set: VersionSet) -> tuple[int, int]:
    return version_set.algo_version_id or -1, version_set.env_version_id


def _version_matches(
    provenance: tuple[ComponentVersionProvenance, ...], refs: tuple[ComponentVersionRef, ...]
) -> bool:
    for ref in refs:
        matches = [
            item
            for item in provenance
            if item.component_name == ref.component_name and item.component_type == ref.component_type
        ]
        if ref.version_id is not None:
            matches = [item for item in matches if item.version_id == ref.version_id]
        if ref.code_snapshot_hash is not None:
            matches = [item for item in matches if item.code_snapshot_hash == ref.code_snapshot_hash]
        if not matches:
            return False
    return True


def select_version_sets(
    candidates: list[RunCandidate], selection: VersionSelection, diagnostics: DiagnosticAccumulator
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
