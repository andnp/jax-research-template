"""Shared data definitions for the experiment dataset boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import polars as pl

from research_analysis.diagnostics.types import DatasetDiagnostic

SweepValue = str | int | float | bool


class SweepValueType(StrEnum):
    """Supported declared sweep-axis value types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class SweepAxis:
    """A typed hyperparameter axis declared by an experiment.

    ``when`` identifies the source hyperparameter values for which the axis applies.
    """

    name: str
    value_type: SweepValueType
    when: Mapping[str, SweepValue] | None = None


@dataclass(frozen=True)
class ExperimentDimensions:
    """Source metadata keys mapped to canonical analysis dimensions."""

    condition: str = "condition_name"
    environment: str | None = None
    sweep_axes: tuple[SweepAxis, ...] = ()


@dataclass(frozen=True)
class ComponentVersionProvenance:
    """Identity and source hash for one component version."""

    component_id: int
    component_name: str
    component_type: str
    version_id: int
    version_number: int
    code_snapshot_hash: str


@dataclass(frozen=True)
class VersionIdSelector:
    """Select a component version by its database row ID."""

    value: int


@dataclass(frozen=True)
class CodeSnapshotSelector:
    """Select a component version by its recorded code snapshot hash."""

    value: str


VersionSelector = VersionIdSelector | CodeSnapshotSelector


@dataclass(frozen=True)
class ComponentVersionRef:
    """Qualified component-version selector."""

    component_name: str
    component_type: str
    selector: VersionSelector

    @classmethod
    def by_id(cls, component_name: str, component_type: str, version_id: int) -> ComponentVersionRef:
        return cls(component_name, component_type, VersionIdSelector(version_id))

    @classmethod
    def by_code_snapshot_hash(
        cls, component_name: str, component_type: str, code_snapshot_hash: str
    ) -> ComponentVersionRef:
        return cls(component_name, component_type, CodeSnapshotSelector(code_snapshot_hash))


@dataclass(frozen=True)
class RunArtifacts:
    """Artifact location and metadata recorded for one completed run."""

    root_path: Path
    manifest_path: Path | None
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ResolvedArtifacts:
    """Validated public artifacts plus the internal metrics database path."""

    reference: RunArtifacts
    metrics_db_path: Path


@dataclass(frozen=True)
class MaterializedRun:
    """One selected run with typed dimensions, provenance, and metric storage."""

    dimension_values: Mapping[str, SweepValue | None]
    seed: int
    run_id: int
    execution_id: int
    ablation: str | None
    provenance: tuple[ComponentVersionProvenance, ...]
    artifacts: RunArtifacts
    metrics_db_path: Path


class VersionSelectionKind(StrEnum):
    """Available component-version selection policies."""

    LATEST_COMPLETED = "latest_completed"
    EXACT = "exact"
    ALL = "all"


@dataclass(frozen=True)
class VersionSelection:
    """Explicit policy for selecting coherent completed version sets."""

    kind: VersionSelectionKind
    exact_versions: tuple[ComponentVersionRef, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is VersionSelectionKind.EXACT and not self.exact_versions:
            raise ValueError("exact version selection requires at least one component version")
        if self.kind is not VersionSelectionKind.EXACT and self.exact_versions:
            raise ValueError("exact_versions are only valid for exact version selection")

    @classmethod
    def latest_completed(cls) -> VersionSelection:
        """Select the newest coherent version set with completed executions."""
        return cls(VersionSelectionKind.LATEST_COMPLETED)

    @classmethod
    def all(cls) -> VersionSelection:
        """Select every completed version set."""
        return cls(VersionSelectionKind.ALL)

    @classmethod
    def exact(cls, versions: tuple[ComponentVersionRef, ...]) -> VersionSelection:
        """Select one qualified historical component-version set."""
        return cls(VersionSelectionKind.EXACT, versions)


@dataclass(frozen=True)
class CohortSelection:
    """Optional dimension filters used to disambiguate a cohort."""

    condition_name: str | None = None
    env_name: str | None = None
    ablation: str | None = None
    sweep_values: tuple[tuple[str, SweepValue], ...] = ()


@dataclass(frozen=True)
class ExperimentDataset:
    """Typed runs, metrics, dimensions, provenance, and loader diagnostics."""

    experiment_name: str
    dimensions: ExperimentDimensions
    runs: pl.DataFrame
    metrics: pl.DataFrame
    provenance: tuple[ComponentVersionProvenance, ...]
    artifacts: Mapping[tuple[int, int], RunArtifacts] = field(default_factory=dict)
    diagnostics: tuple[DatasetDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        required_runs = {
            "condition_name",
            "seed",
            "run_id",
            "execution_id",
            "ablation",
        }
        required_metrics = {
            "condition_name",
            "seed",
            "run_id",
            "execution_id",
            "step",
            "metric",
            "value",
        }
        missing_runs = required_runs.difference(self.runs.columns)
        missing_metrics = required_metrics.difference(self.metrics.columns)
        if missing_runs:
            raise TypeError(f"runs table is missing columns: {sorted(missing_runs)}")
        if missing_metrics:
            raise TypeError(f"metrics table is missing columns: {sorted(missing_metrics)}")
