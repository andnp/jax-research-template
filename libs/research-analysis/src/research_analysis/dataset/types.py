"""Shared data definitions for the experiment dataset boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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

    ``when`` maps source hyperparameters to the values under which the axis applies: it applies
    to a run whose value for every key is one of that key's listed values, and is null elsewhere.
    """

    name: str
    value_type: SweepValueType
    when: Mapping[str, tuple[SweepValue, ...]] | None = None

    def __post_init__(self) -> None:
        for key, values in (self.when or {}).items():
            if not isinstance(values, tuple) or not values or len(set(values)) < len(values):
                raise ValueError(
                    f"sweep axis {self.name!r}: when[{key!r}] must be a non-empty tuple of distinct values, "
                    f"e.g. ('adversarial',), got {values!r}"
                )


@dataclass(frozen=True)
class ExperimentDimensions:
    """Source metadata keys mapped to canonical analysis dimensions."""

    condition: str = "condition_name"
    environment: str | None = None
    sweep_axes: tuple[SweepAxis, ...] = ()

    def __post_init__(self) -> None:
        if not self.condition.strip():
            raise ValueError("condition must name a hyperparameter")
        if self.environment is not None and not self.environment.strip():
            raise ValueError("environment must name a hyperparameter")


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
class ExecutionProvenance:
    """Which code and host produced one included execution."""

    execution_id: int
    git_commit: str | None
    # Uncommitted changes were recorded, so git_commit alone does not reproduce this execution.
    dirty: bool
    hostname: str | None
    start_time: str | None
    end_time: str | None


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


CellKey = tuple[tuple[str, SweepValue | None], ...]

_SCALAR_METRIC_SCHEMA = {
    "condition_name": pl.String,
    "seed": pl.Int64,
    "run_id": pl.Int64,
    "execution_id": pl.Int64,
    "ablation": pl.String,
    "metric": pl.String,
    "value": pl.Float64,
}
_EXECUTION_METRIC_SCHEMA = {"execution_id": pl.Int64, "metric": pl.String, "value": pl.Float64}


class ExclusionReason(StrEnum):
    """Why a planned run is absent from the dataset's runs table."""

    VERSION_NOT_SELECTED = "version_not_selected"
    MALFORMED_ARTIFACT = "malformed_artifact"
    MISSING_METRIC_ARTIFACT = "missing_metric_artifact"
    UNRESOLVED_DIMENSION = "unresolved_dimension"
    MISSING_HYPER_CONFIG = "missing_hyper_config"


@dataclass(frozen=True)
class CellAccounting:
    """Outcome counts for every planned run of one cell; diverged runs are also counted as completed."""

    cell: CellKey
    planned: int
    completed: int
    failed: int
    pending: int
    diverged: Mapping[str, int]
    excluded: Mapping[ExclusionReason, int]


@dataclass(frozen=True)
class RunAccounting:
    """Per-cell run outcomes plus planned runs that could not be placed in any cell."""

    cells: tuple[CellAccounting, ...] = ()
    unassigned: Mapping[ExclusionReason, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentDataset:
    """Typed runs, curve and scalar metrics, dimensions, provenance, and diagnostics."""

    experiment_name: str
    dimensions: ExperimentDimensions
    runs: pl.DataFrame
    metrics: pl.DataFrame
    provenance: tuple[ComponentVersionProvenance, ...]
    artifacts: Mapping[tuple[int, int], RunArtifacts] = field(default_factory=dict)
    diagnostics: tuple[DatasetDiagnostic, ...] = ()
    accounting: RunAccounting = field(default_factory=RunAccounting)
    executions: tuple[ExecutionProvenance, ...] = ()
    scalar_metrics: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=_SCALAR_METRIC_SCHEMA))
    execution_metrics: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=_EXECUTION_METRIC_SCHEMA))

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
        required_scalar_metrics = set(_SCALAR_METRIC_SCHEMA)
        required_execution_metrics = set(_EXECUTION_METRIC_SCHEMA)
        missing_runs = required_runs.difference(self.runs.columns)
        missing_metrics = required_metrics.difference(self.metrics.columns)
        missing_scalar_metrics = required_scalar_metrics.difference(self.scalar_metrics.columns)
        missing_execution_metrics = required_execution_metrics.difference(self.execution_metrics.columns)
        if missing_runs:
            raise TypeError(f"runs table is missing columns: {sorted(missing_runs)}")
        if missing_metrics:
            raise TypeError(f"metrics table is missing columns: {sorted(missing_metrics)}")
        if missing_scalar_metrics:
            raise TypeError(f"scalar_metrics table is missing columns: {sorted(missing_scalar_metrics)}")
        if missing_execution_metrics:
            raise TypeError(f"execution_metrics table is missing columns: {sorted(missing_execution_metrics)}")

    def filter_runs(self, predicate: pl.Expr) -> ExperimentDataset:
        """Restrict to runs matching ``predicate``, keeping every run-keyed frame consistent with them.

        An execution metric is a whole-execution value, kept while any of its runs survives; diagnostics,
        accounting, provenance, and executions still describe the whole load.
        """
        runs = self.runs.filter(predicate)
        run_keys = runs.select("run_id", "execution_id").unique()
        kept = set(run_keys.iter_rows())
        return replace(
            self,
            runs=runs,
            metrics=self.metrics.join(run_keys, on=["run_id", "execution_id"], how="semi", maintain_order="left"),
            scalar_metrics=self.scalar_metrics.join(
                run_keys, on=["run_id", "execution_id"], how="semi", maintain_order="left"
            ),
            execution_metrics=self.execution_metrics.join(
                run_keys.select("execution_id").unique(), on="execution_id", how="semi", maintain_order="left"
            ),
            artifacts={key: value for key, value in self.artifacts.items() if key in kept},
        )
