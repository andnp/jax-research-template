"""Resolve which runs, executions, and version sets exist for an experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from experiment_definition.db import DatabaseManager
from experiment_definition.records import (
    ExecutionArtifactRow,
    ExecutionRow,
    ExperimentRow,
    RunRow,
)
from pydantic import BaseModel, ConfigDict

from research_analysis.dataset.types import ComponentVersionProvenance, RunArtifacts
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode
from research_analysis.validation import validate_json


class VersionSet(NamedTuple):
    """Identifies one coherent algorithm/environment component-version pairing."""

    algo_version_id: int | None
    env_version_id: int | None


class _ArtifactMetadata(BaseModel):
    """Required shape of an execution artifact's recorded metadata."""

    model_config = ConfigDict(extra="allow")

    metrics_db_path: str


@dataclass(frozen=True)
class RunCandidate:
    """One run considered for dataset loading, with its resolved version identity, provenance, and artifact."""

    run: RunRow
    execution: ExecutionRow
    artifacts: RunArtifacts
    version_set: VersionSet
    provenance: tuple[ComponentVersionProvenance, ...]


def _resolve_artifact(
    artifact: ExecutionArtifactRow | None, run: RunRow, execution: ExecutionRow, diagnostics: DiagnosticAccumulator
) -> RunArtifacts | None:
    if artifact is None:
        diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "completed run has no artifact record", run.id, execution.id)
        return None
    metadata = validate_json(_ArtifactMetadata, artifact.metadata_json or "{}")
    if metadata is None:
        diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "completed run has no recorded metrics_db_path", run.id, execution.id)
        return None
    run_artifacts = RunArtifacts(
        root_path=Path(artifact.root_path),
        manifest_path=Path(artifact.manifest_path) if artifact.manifest_path is not None else None,
        metrics_db_path=metadata.metrics_db_path,
        metadata=metadata.model_dump(),
    )
    return run_artifacts


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


def fetch_run_candidates(database: DatabaseManager, experiment: ExperimentRow, diagnostics: DiagnosticAccumulator) -> list[RunCandidate]:
    provenance_by_version: dict[int, ComponentVersionProvenance] = {}
    candidates: list[RunCandidate] = []
    for run in database.list_runs(experiment.id):
        execution = database.get_latest_completed_execution_for_run(run.id)
        if execution is None:
            diagnostics.add(DiagnosticCode.INCOMPLETE_EXECUTION, "run has no completed execution", run.id)
            continue
        artifacts = _resolve_artifact(database.get_latest_completed_artifacts_for_run(run.id), run, execution, diagnostics)
        if artifacts is None:
            continue
        version_items = tuple(
            _cached_component_provenance(database, version_id, provenance_by_version)
            for version_id in (run.algo_version_id, run.env_version_id)
            if version_id is not None
        )
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
                artifacts=artifacts,
                version_set=VersionSet(run.algo_version_id, run.env_version_id),
                provenance=version_items,
            )
        )
    return candidates
