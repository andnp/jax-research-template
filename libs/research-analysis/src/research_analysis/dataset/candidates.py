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
from pydantic import BaseModel, ConfigDict

from research_analysis.dataset.types import RunArtifacts
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
    artifacts: RunArtifacts


def _resolve_artifact(artifact: ExecutionArtifactRow | None) -> RunArtifacts | None:
    if artifact is None:
        return None
    metadata = validate_json(_ArtifactMetadata, artifact.metadata_json or "{}")
    if metadata is None:
        return None
    run_artifacts = RunArtifacts(
        root_path=Path(artifact.root_path),
        manifest_path=Path(artifact.manifest_path) if artifact.manifest_path is not None else None,
        metrics_db_path=metadata.metrics_db_path,
        metadata=metadata.model_dump(),
    )
    return run_artifacts


def fetch_completed_runs(database: DatabaseManager, experiment: ExperimentRow) -> list[CompletedRun]:
    completed_runs: list[CompletedRun] = []
    for run in database.list_runs(experiment.id):
        execution = database.get_latest_completed_execution_for_run(run.id)
        if execution is None:
            continue
        artifacts = _resolve_artifact(database.get_execution_artifacts(execution.id))
        if artifacts is None:
            continue
        completed_runs.append(CompletedRun(run=run, execution=execution, artifacts=artifacts))
    return completed_runs
