"""Resolve which runs, executions, and version sets exist for an experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from experiment_definition.db import (
    DatabaseManager,
    ExecutionArtifactRow,
    ExecutionRow,
    ExperimentRow,
    RunRow,
)

from research_analysis.dataset.types import ComponentVersionProvenance
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


class VersionSet(NamedTuple):
    """Identifies one coherent algorithm/environment component-version pairing."""

    algo_version_id: int | None
    env_version_id: int | None


@dataclass(frozen=True)
class RunCandidate:
    """One run considered for dataset loading, with its resolved version identity and provenance."""

    run: RunRow
    execution: ExecutionRow
    artifact: ExecutionArtifactRow | None
    version_set: VersionSet
    provenance: tuple[ComponentVersionProvenance, ...]


def _component_provenance(database: DatabaseManager, version_id: int) -> ComponentVersionProvenance | None:
    row = database.conn.execute(
        "SELECT c.id, c.name, c.type, cv.id, cv.version_number, cv.code_snapshot_hash "
        "FROM ComponentVersions cv INNER JOIN Components c ON c.id = cv.component_id "
        "WHERE cv.id = ?",
        (version_id,),
    ).fetchone()
    if row is None:
        return None
    return ComponentVersionProvenance(
        component_id=int(row[0]),
        component_name=str(row[1]),
        component_type=str(row[2]),
        version_id=int(row[3]),
        version_number=int(row[4]),
        code_snapshot_hash=str(row[5]),
    )


def fetch_run_candidates(database: DatabaseManager, experiment: ExperimentRow, diagnostics: DiagnosticAccumulator) -> list[RunCandidate]:
    provenance_by_version: dict[int, ComponentVersionProvenance] = {}
    candidates: list[RunCandidate] = []
    for run in database.list_runs(experiment.id):
        execution = database.get_latest_completed_execution_for_run(run.id)
        if execution is None:
            diagnostics.add(DiagnosticCode.INCOMPLETE_EXECUTION, "run has no completed execution", run.id)
            continue
        artifact = database.get_latest_completed_artifacts_for_run(run.id)
        version_items: list[ComponentVersionProvenance] = []
        for version_id in (run.algo_version_id, run.env_version_id):
            if version_id is None:
                continue
            item = provenance_by_version.get(version_id)
            if item is None:
                item = _component_provenance(database, version_id)
                if item is not None:
                    provenance_by_version[version_id] = item
            if item is not None:
                version_items.append(item)
        if run.algo_version_id is None:
            diagnostics.add(
                DiagnosticCode.UNKNOWN_COMPONENT_PROVENANCE,
                "run has NULL algorithm provenance because multiple ALGO components are declared",
                run.id,
                execution.id,
            )
        candidates.append(
            RunCandidate(
                run=run,
                execution=execution,
                artifact=artifact,
                version_set=VersionSet(run.algo_version_id, run.env_version_id),
                provenance=tuple(version_items),
            )
        )
    return candidates
