"""Resolve which runs, executions, and version sets exist for an experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment_definition.db import DatabaseManager
from experiment_definition.records import (
    ExecutionArtifactRow,
    ExecutionRow,
    ExperimentRow,
    RunRow,
)
from experiment_definition.status import RunState, run_state
from pydantic import BaseModel, ConfigDict

from research_analysis.dataset.types import ResolvedArtifacts, RunArtifacts
from research_analysis.validation import validate_json


class _ArtifactMetadata(BaseModel):
    """Required shape of an execution artifact's recorded metadata."""

    model_config = ConfigDict(extra="allow")

    metrics_db_path: str


@dataclass(frozen=True)
class CompletedRun:
    """One run with a completed execution and usable recorded artifacts."""

    run: RunRow
    execution: ExecutionRow
    artifacts: ResolvedArtifacts


def _resolve_artifact(artifact: ExecutionArtifactRow | None) -> ResolvedArtifacts | None:
    if artifact is None:
        return None
    metadata = validate_json(_ArtifactMetadata, artifact.metadata_json or "{}")
    if metadata is None:
        return None
    metadata_values = metadata.model_dump()
    metadata_values.pop("metrics_db_path")
    return ResolvedArtifacts(
        reference=RunArtifacts(
            root_path=Path(artifact.root_path),
            manifest_path=Path(artifact.manifest_path) if artifact.manifest_path is not None else None,
            metadata=metadata_values,
        ),
        metrics_db_path=Path(metadata.metrics_db_path),
    )


@dataclass(frozen=True)
class RunCandidate:
    """One planned run, its execution outcome, and its usable completed execution if any."""

    run: RunRow
    state: RunState
    completed: CompletedRun | None


def _completed_run(database: DatabaseManager, run: RunRow) -> CompletedRun | None:
    execution = database.get_latest_completed_execution_for_run(run.id)
    if execution is None:
        return None
    artifacts = _resolve_artifact(database.get_execution_artifacts(execution.id))
    if artifacts is None:
        return None
    return CompletedRun(run=run, execution=execution, artifacts=artifacts)


def fetch_run_candidates(database: DatabaseManager, experiment: ExperimentRow) -> list[RunCandidate]:
    statuses = database.list_run_execution_statuses(experiment.id)
    candidates: list[RunCandidate] = []
    for run in database.list_runs(experiment.id):
        state = run_state(statuses.get(run.id, frozenset()))
        completed = _completed_run(database, run) if state is RunState.COMPLETED else None
        candidates.append(RunCandidate(run=run, state=state, completed=completed))
    return candidates
