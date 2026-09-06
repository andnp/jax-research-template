"""Canonical experiment dataset loading for analysis workflows."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl
from experiment_definition.db import (
    DatabaseManager,
    ExecutionArtifactRow,
    ExecutionRow,
    RunRow,
)

from research_analysis.loader import load_sqlite_query

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
class ComponentVersionRef:
    """Qualified component-version selector."""

    component_name: str
    component_type: str
    version_id: int | None = None
    code_snapshot_hash: str | None = None

    def __post_init__(self) -> None:
        if (self.version_id is None) == (self.code_snapshot_hash is None):
            raise ValueError("exact component versions require exactly one version_id or code_snapshot_hash")


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


class DiagnosticCode(StrEnum):
    """Structured dataset-loading diagnostic categories."""

    INCOMPLETE_EXECUTION = "incomplete_execution"
    MISSING_METRIC_ARTIFACT = "missing_metric_artifact"
    EXCLUDED_VERSION = "excluded_version"
    AMBIGUOUS_COHORT = "ambiguous_cohort"
    UNKNOWN_COMPONENT_PROVENANCE = "unknown_component_provenance"
    MISSING_DIMENSION = "missing_dimension"
    INVALID_DIMENSION = "invalid_dimension"
    INVALID_VERSION_SELECTION = "invalid_version_selection"


@dataclass(frozen=True)
class DatasetDiagnostic:
    """A structured issue encountered while selecting or materializing data."""

    code: DiagnosticCode
    message: str
    run_id: int | None = None
    execution_id: int | None = None


class DatasetLoadError(ValueError):
    """Raised when strict dataset loading encounters a fatal diagnostic."""

    def __init__(self, diagnostics: tuple[DatasetDiagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__("experiment dataset loading failed: " + "; ".join(d.message for d in diagnostics))


@dataclass(frozen=True)
class ExperimentDataset:
    """Typed runs, metrics, dimensions, provenance, and loader diagnostics."""

    experiment_name: str
    dimensions: ExperimentDimensions
    runs: pl.DataFrame
    metrics: pl.DataFrame
    provenance: tuple[ComponentVersionProvenance, ...]
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


_FATAL_CODES = frozenset(
    {
        DiagnosticCode.AMBIGUOUS_COHORT,
        DiagnosticCode.MISSING_METRIC_ARTIFACT,
        DiagnosticCode.MISSING_DIMENSION,
        DiagnosticCode.INVALID_DIMENSION,
        DiagnosticCode.INVALID_VERSION_SELECTION,
    }
)


def _diagnostic(
    code: DiagnosticCode,
    message: str,
    run_id: int | None = None,
    execution_id: int | None = None,
) -> DatasetDiagnostic:
    return DatasetDiagnostic(code, message, run_id, execution_id)


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


def _coerce_axis_value(value: object, axis: SweepAxis) -> SweepValue | None:
    if axis.value_type is SweepValueType.STRING and isinstance(value, str):
        return value
    if axis.value_type is SweepValueType.INTEGER and isinstance(value, int) and not isinstance(value, bool):
        return value
    if axis.value_type is SweepValueType.FLOAT and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if axis.value_type is SweepValueType.BOOLEAN and isinstance(value, bool):
        return value
    return None


def _axis_dtype(axis: SweepAxis):
    if axis.value_type is SweepValueType.STRING:
        return pl.String
    if axis.value_type is SweepValueType.INTEGER:
        return pl.Int64
    if axis.value_type is SweepValueType.FLOAT:
        return pl.Float64
    return pl.Boolean


def _axis_applicability(axis: SweepAxis, hyperparameters: dict[str, object]) -> bool | None:
    if axis.when is None:
        return True
    for key, value in axis.when.items():
        if key not in hyperparameters:
            return None
        if hyperparameters[key] != value:
            return False
    return True


def _version_key(algo_version_id: int | None, env_version_id: int | None) -> tuple[int | None, int | None]:
    return algo_version_id, env_version_id


def _version_sort_key(version_set: tuple[int | None, int | None]) -> tuple[int, int]:
    return version_set[0] or -1, version_set[1] or -1


def _required_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected integer dataset identity, got {value!r}")
    return value


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


def _cohort_matches(row: dict[str, object], selection: CohortSelection) -> bool:
    if selection.condition_name is not None and row.get("condition_name") != selection.condition_name:
        return False
    if selection.env_name is not None and row.get("env_name") != selection.env_name:
        return False
    if selection.ablation is not None and row.get("ablation") != selection.ablation:
        return False
    return all(row.get(key) == value for key, value in selection.sweep_values)


def _metric_query_pairs(pairs: list[tuple[int, int]], metrics: tuple[str, ...] | None) -> str:
    pair_sql = " OR ".join(f"(run_id = {run_id} AND execution_id = {execution_id})" for run_id, execution_id in pairs)
    metric_sql = ""
    if metrics is not None:
        quoted = ", ".join("'" + metric.replace("'", "''") + "'" for metric in metrics)
        metric_sql = f" AND metric_name IN ({quoted})"
    return (
        "SELECT run_id, execution_id, global_step, metric_name, value FROM metrics "
        f"WHERE ({pair_sql}){metric_sql} ORDER BY run_id, execution_id, metric_name, global_step"
    )


def load_experiment_dataset(
    experiments_db: Path | str,
    slug: str,
    *,
    dimensions: ExperimentDimensions | None = None,
    version_selection: VersionSelection | None = None,
    cohort: CohortSelection | None = None,
    metrics: tuple[str, ...] | None = None,
    partial: bool = False,
) -> ExperimentDataset:
    """Load one typed experiment dataset from its recorded artifacts."""
    selected_dimensions = dimensions or ExperimentDimensions()
    selected_versions = version_selection or VersionSelection.latest_completed()
    diagnostics: list[DatasetDiagnostic] = []
    rows: list[dict[str, object]] = []
    provenance_by_version: dict[int, ComponentVersionProvenance] = {}
    candidates: list[
        tuple[RunRow, ExecutionRow, ExecutionArtifactRow | None, tuple[int | None, int | None], tuple[ComponentVersionProvenance, ...]]
    ] = []

    with DatabaseManager(Path(experiments_db)) as database:
        database.initialize()
        experiment = database.get_experiment(slug)
        if experiment is None:
            raise ValueError(f"Unknown experiment slug {slug!r} in {experiments_db}.")

        for run in database.list_runs(experiment.id):
            execution = database.get_latest_completed_execution_for_run(run.id)
            if execution is None:
                diagnostics.append(
                    _diagnostic(DiagnosticCode.INCOMPLETE_EXECUTION, "run has no completed execution", run.id)
                )
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
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.UNKNOWN_COMPONENT_PROVENANCE,
                        "run has NULL algorithm provenance because multiple ALGO components are declared",
                        run.id,
                        execution.id,
                    )
                )
            candidates.append((run, execution, artifact, _version_key(run.algo_version_id, run.env_version_id), tuple(version_items)))

        available_sets = {item[3] for item in candidates}
        selected_sets = available_sets
        if selected_versions.kind is VersionSelectionKind.LATEST_COMPLETED and available_sets:
            selected_sets = {max(available_sets, key=_version_sort_key)}
        elif selected_versions.kind is VersionSelectionKind.EXACT:
            selected_sets = {
                item[3]
                for item in candidates
                if _version_matches(item[4], selected_versions.exact_versions)
            }
            if not selected_sets:
                diagnostics.append(_diagnostic(DiagnosticCode.INVALID_VERSION_SELECTION, "exact component versions have no completed runs"))

        for version_set in sorted(available_sets, key=_version_sort_key):
            if version_set not in selected_sets:
                diagnostics.append(
                    _diagnostic(DiagnosticCode.EXCLUDED_VERSION, f"excluded completed version set {version_set!r}")
                )

        all_provenance: dict[int, ComponentVersionProvenance] = {}
        for _, _, _, version_set, items in candidates:
            if version_set in selected_sets:
                all_provenance.update({item.version_id: item for item in items})

        for run, execution, artifact, version_set, items in candidates:
            if version_set not in selected_sets:
                continue
            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                diagnostics.append(_diagnostic(DiagnosticCode.MISSING_DIMENSION, "run has no hyperparameter configuration", run.id, execution.id))
                continue
            hyperparameters = json.loads(hyper_config.json_blob)
            if not isinstance(hyperparameters, dict):
                diagnostics.append(_diagnostic(DiagnosticCode.MISSING_DIMENSION, "hyperparameter configuration is not an object", run.id, execution.id))
                continue
            condition_value = hyperparameters.get(selected_dimensions.condition)
            if condition_value is None:
                diagnostics.append(_diagnostic(DiagnosticCode.MISSING_DIMENSION, f"missing condition key {selected_dimensions.condition!r}", run.id, execution.id))
                continue
            row: dict[str, object] = {
                "condition_name": str(condition_value),
                "seed": run.seed,
                "run_id": run.id,
                "execution_id": execution.id,
                "ablation": run.ablation,
                "metrics_db_path": None,
            }
            if selected_dimensions.environment is not None:
                environment_value = hyperparameters.get(selected_dimensions.environment)
                if environment_value is None:
                    diagnostics.append(_diagnostic(DiagnosticCode.MISSING_DIMENSION, f"missing environment key {selected_dimensions.environment!r}", run.id, execution.id))
                    continue
                row["env_name"] = str(environment_value)
            for axis in selected_dimensions.sweep_axes:
                applies = _axis_applicability(axis, hyperparameters)
                if applies is None:
                    diagnostics.append(
                        _diagnostic(
                            DiagnosticCode.INVALID_DIMENSION,
                            f"cannot evaluate applicability for sweep axis {axis.name!r}",
                            run.id,
                            execution.id,
                        )
                    )
                    break
                if not applies:
                    if axis.name in hyperparameters:
                        diagnostics.append(
                            _diagnostic(
                                DiagnosticCode.INVALID_DIMENSION,
                                f"sweep axis {axis.name!r} is present outside its applicability condition",
                                run.id,
                                execution.id,
                            )
                        )
                        break
                    row[axis.name] = None
                    continue
                if axis.name not in hyperparameters:
                    diagnostics.append(
                        _diagnostic(
                            DiagnosticCode.MISSING_DIMENSION,
                            f"missing sweep axis {axis.name!r} for an applicable run",
                            run.id,
                            execution.id,
                        )
                    )
                    break
                raw_value = hyperparameters[axis.name]
                value = _coerce_axis_value(raw_value, axis)
                if value is None:
                    diagnostics.append(_diagnostic(DiagnosticCode.INVALID_DIMENSION, f"invalid value for sweep axis {axis.name!r}", run.id, execution.id))
                    break
                row[axis.name] = value
            else:
                metadata: object = json.loads(artifact.metadata_json) if artifact is not None and artifact.metadata_json else {}
                if not isinstance(metadata, dict) or not isinstance(metadata.get("metrics_db_path"), str):
                    diagnostics.append(_diagnostic(DiagnosticCode.MISSING_METRIC_ARTIFACT, "completed run has no recorded metrics_db_path", run.id, execution.id))
                    continue
                row["metrics_db_path"] = metadata["metrics_db_path"]
                if items:
                    for item in items:
                        prefix = "algo" if item.component_type == "ALGO" else "env"
                        row[f"{prefix}_component_name"] = item.component_name
                        row[f"{prefix}_component_type"] = item.component_type
                        row[f"{prefix}_version_id"] = item.version_id
                        row[f"{prefix}_version_number"] = item.version_number
                        row[f"{prefix}_code_snapshot_hash"] = item.code_snapshot_hash
                if cohort is None or _cohort_matches(row, cohort):
                    rows.append(row)

        metric_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        run_rows = {
            (_required_int(row["run_id"]), _required_int(row["execution_id"])): row
            for row in rows
        }
        for pair, row in run_rows.items():
            path = row["metrics_db_path"]
            if isinstance(path, str):
                metric_groups[path].append(pair)

        metric_frames: list[pl.DataFrame] = []
        for path, pairs in metric_groups.items():
            metric_path = Path(path)
            if not metric_path.exists():
                for run_id, execution_id in pairs:
                    diagnostics.append(_diagnostic(DiagnosticCode.MISSING_METRIC_ARTIFACT, f"metrics database does not exist: {metric_path}", run_id, execution_id))
                continue
            frame = load_sqlite_query(metric_path, _metric_query_pairs(pairs, metrics))
            if frame.is_empty():
                for run_id, execution_id in pairs:
                    diagnostics.append(_diagnostic(DiagnosticCode.MISSING_METRIC_ARTIFACT, "metrics database has no requested rows", run_id, execution_id))
                continue
            metadata = pl.DataFrame([run_rows[pair] for pair in pairs])
            metric_frames.append(
                frame.join(metadata, on=["run_id", "execution_id"], how="inner")
                .rename({"global_step": "step", "metric_name": "metric"})
                .select(
                    [
                        "condition_name",
                        *(["env_name"] if "env_name" in metadata.columns else []),
                        "seed",
                        "run_id",
                        "execution_id",
                        "ablation",
                        "step",
                        "metric",
                        "value",
                    ]
                )
            )

        if rows:
            run_frame = pl.DataFrame(
                rows,
                schema_overrides={axis.name: _axis_dtype(axis) for axis in selected_dimensions.sweep_axes},
            )
        else:
            run_schema = {
                "condition_name": pl.String,
                **({"env_name": pl.String} if selected_dimensions.environment is not None else {}),
                "seed": pl.Int64,
                "run_id": pl.Int64,
                "execution_id": pl.Int64,
                "ablation": pl.String,
                "metrics_db_path": pl.String,
                **{axis.name: _axis_dtype(axis) for axis in selected_dimensions.sweep_axes},
            }
            run_frame = pl.DataFrame(schema=run_schema)
        metric_frame = pl.concat(metric_frames).sort(["condition_name", "seed", "step", "metric"]) if metric_frames else pl.DataFrame(
            schema={
                "condition_name": pl.String,
                **({"env_name": pl.String} if selected_dimensions.environment is not None else {}),
                "seed": pl.Int64,
                "run_id": pl.Int64,
                "execution_id": pl.Int64,
                "ablation": pl.String,
                "step": pl.Int64,
                "metric": pl.String,
                "value": pl.Float64,
            }
        )

    fatal = tuple(item for item in diagnostics if item.code in _FATAL_CODES)
    if fatal and not partial:
        raise DatasetLoadError(fatal)
    return ExperimentDataset(
        experiment_name=experiment.name,
        dimensions=selected_dimensions,
        runs=run_frame,
        metrics=metric_frame,
        provenance=tuple(all_provenance.values()),
        diagnostics=tuple(diagnostics),
    )
