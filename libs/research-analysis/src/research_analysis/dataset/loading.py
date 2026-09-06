"""Canonical experiment dataset loading for analysis workflows."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from experiment_definition.db import (
    DatabaseManager,
    ExecutionArtifactRow,
    ExecutionRow,
    RunRow,
)

from research_analysis.dataset.candidates import RunCandidate, fetch_run_candidates
from research_analysis.dataset.types import (
    CohortSelection,
    ComponentVersionProvenance,
    ComponentVersionRef,
    ExperimentDataset,
    ExperimentDimensions,
    RunArtifacts,
    SweepAxis,
    SweepValue,
    SweepValueType,
    VersionSelection,
    VersionSelectionKind,
)
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import FATAL_DIAGNOSTIC_CODES, DatasetLoadError, DiagnosticCode
from research_analysis.loader import load_sqlite_query


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


def _axis_applicability(axis: SweepAxis, hyperparameters: Mapping[str, object]) -> bool | None:
    if axis.when is None:
        return True
    for key, value in axis.when.items():
        if key not in hyperparameters:
            return None
        if hyperparameters[key] != value:
            return False
    return True


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


def _select_version_sets(
    candidates: list[RunCandidate], selection: VersionSelection, diagnostics: DiagnosticAccumulator
) -> tuple[frozenset[tuple[int | None, int | None]], dict[int, ComponentVersionProvenance]]:
    available_sets = {item[3] for item in candidates}
    selected_sets = available_sets
    if selection.kind is VersionSelectionKind.LATEST_COMPLETED and available_sets:
        selected_sets = {max(available_sets, key=_version_sort_key)}
    elif selection.kind is VersionSelectionKind.EXACT:
        selected_sets = {item[3] for item in candidates if _version_matches(item[4], selection.exact_versions)}
        if not selected_sets:
            diagnostics.add(DiagnosticCode.INVALID_VERSION_SELECTION, "exact component versions have no completed runs")

    for version_set in sorted(available_sets, key=_version_sort_key):
        if version_set not in selected_sets:
            diagnostics.add(DiagnosticCode.EXCLUDED_VERSION, f"excluded completed version set {version_set!r}")

    all_provenance: dict[int, ComponentVersionProvenance] = {}
    for _, _, _, version_set, items in candidates:
        if version_set in selected_sets:
            all_provenance.update({item.version_id: item for item in items})

    return frozenset(selected_sets), all_provenance


def _cohort_matches(row: dict[str, object], selection: CohortSelection) -> bool:
    if selection.condition_name is not None and row.get("condition_name") != selection.condition_name:
        return False
    if selection.env_name is not None and row.get("env_name") != selection.env_name:
        return False
    if selection.ablation is not None and row.get("ablation") != selection.ablation:
        return False
    return all(row.get(key) == value for key, value in selection.sweep_values)


def _build_run_row(
    run: RunRow,
    execution: ExecutionRow,
    artifact: ExecutionArtifactRow | None,
    items: tuple[ComponentVersionProvenance, ...],
    hyperparameters: Mapping[str, object],
    dimensions: ExperimentDimensions,
    diagnostics: DiagnosticAccumulator,
) -> tuple[dict[str, object], RunArtifacts] | None:
    condition_value = hyperparameters.get(dimensions.condition)
    if condition_value is None:
        diagnostics.add(DiagnosticCode.MISSING_DIMENSION, f"missing condition key {dimensions.condition!r}", run.id, execution.id)
        return None
    row: dict[str, object] = {
        "condition_name": str(condition_value),
        "seed": run.seed,
        "run_id": run.id,
        "execution_id": execution.id,
        "ablation": run.ablation,
        "metrics_db_path": None,
    }
    if dimensions.environment is not None:
        environment_value = hyperparameters.get(dimensions.environment)
        if environment_value is None:
            diagnostics.add(
                DiagnosticCode.MISSING_DIMENSION, f"missing environment key {dimensions.environment!r}", run.id, execution.id
            )
            return None
        row["env_name"] = str(environment_value)
    for axis in dimensions.sweep_axes:
        applies = _axis_applicability(axis, hyperparameters)
        if applies is None:
            diagnostics.add(
                DiagnosticCode.INVALID_DIMENSION,
                f"cannot evaluate applicability for sweep axis {axis.name!r}",
                run.id,
                execution.id,
            )
            return None
        if not applies:
            if axis.name in hyperparameters:
                diagnostics.add(
                    DiagnosticCode.INVALID_DIMENSION,
                    f"sweep axis {axis.name!r} is present outside its applicability condition",
                    run.id,
                    execution.id,
                )
                return None
            row[axis.name] = None
            continue
        if axis.name not in hyperparameters:
            diagnostics.add(
                DiagnosticCode.MISSING_DIMENSION,
                f"missing sweep axis {axis.name!r} for an applicable run",
                run.id,
                execution.id,
            )
            return None
        raw_value = hyperparameters[axis.name]
        value = _coerce_axis_value(raw_value, axis)
        if value is None:
            diagnostics.add(DiagnosticCode.INVALID_DIMENSION, f"invalid value for sweep axis {axis.name!r}", run.id, execution.id)
            return None
        row[axis.name] = value
    if artifact is None:
        diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "completed run has no artifact record", run.id, execution.id)
        return None
    metadata: object = json.loads(artifact.metadata_json) if artifact.metadata_json else {}
    if not isinstance(metadata, dict) or not isinstance(metadata.get("metrics_db_path"), str):
        diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "completed run has no recorded metrics_db_path", run.id, execution.id)
        return None
    row["metrics_db_path"] = metadata["metrics_db_path"]
    for item in items:
        prefix = "algo" if item.component_type == "ALGO" else "env"
        row[f"{prefix}_component_name"] = item.component_name
        row[f"{prefix}_component_type"] = item.component_type
        row[f"{prefix}_version_id"] = item.version_id
        row[f"{prefix}_version_number"] = item.version_number
        row[f"{prefix}_code_snapshot_hash"] = item.code_snapshot_hash
    return row, RunArtifacts(
        root_path=Path(artifact.root_path),
        manifest_path=Path(artifact.manifest_path) if artifact.manifest_path is not None else None,
        metadata=metadata,
    )


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


def _load_metric_frames(
    rows: list[dict[str, object]], metrics: tuple[str, ...] | None, diagnostics: DiagnosticAccumulator
) -> list[pl.DataFrame]:
    metric_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    run_rows = {(_required_int(row["run_id"]), _required_int(row["execution_id"])): row for row in rows}
    for pair, row in run_rows.items():
        path = row["metrics_db_path"]
        if isinstance(path, str):
            metric_groups[path].append(pair)

    metric_frames: list[pl.DataFrame] = []
    for path, pairs in metric_groups.items():
        metric_path = Path(path)
        if not metric_path.exists():
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, f"metrics database does not exist: {metric_path}", run_id, execution_id)
            continue
        frame = load_sqlite_query(metric_path, _metric_query_pairs(pairs, metrics))
        if frame.is_empty():
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "metrics database has no requested rows", run_id, execution_id)
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
    return metric_frames


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
    diagnostics = DiagnosticAccumulator()
    rows: list[dict[str, object]] = []
    artifacts: dict[tuple[int, int], RunArtifacts] = {}

    with DatabaseManager(Path(experiments_db)) as database:
        database.initialize()
        experiment = database.get_experiment(slug)
        if experiment is None:
            raise ValueError(f"Unknown experiment slug {slug!r} in {experiments_db}.")

        candidates = fetch_run_candidates(database, experiment, diagnostics)
        selected_sets, all_provenance = _select_version_sets(candidates, selected_versions, diagnostics)

        for run, execution, artifact, version_set, items in candidates:
            if version_set not in selected_sets:
                continue
            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                diagnostics.add(DiagnosticCode.MISSING_DIMENSION, "run has no hyperparameter configuration", run.id, execution.id)
                continue
            hyperparameters = hyper_config.hyperparameters
            if not isinstance(hyperparameters, dict):
                diagnostics.add(DiagnosticCode.MISSING_DIMENSION, "hyperparameter configuration is not an object", run.id, execution.id)
                continue
            built = _build_run_row(run, execution, artifact, items, hyperparameters, selected_dimensions, diagnostics)
            if built is not None and (cohort is None or _cohort_matches(built[0], cohort)):
                row, run_artifacts = built
                rows.append(row)
                artifacts[(run.id, execution.id)] = run_artifacts

    metric_frames = _load_metric_frames(rows, metrics, diagnostics)

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

    fatal = diagnostics.fatal(FATAL_DIAGNOSTIC_CODES)
    if fatal and not partial:
        raise DatasetLoadError(fatal)
    return ExperimentDataset(
        experiment_name=experiment.name,
        dimensions=selected_dimensions,
        runs=run_frame,
        metrics=metric_frame,
        provenance=tuple(all_provenance.values()),
        artifacts=artifacts,
        diagnostics=diagnostics.as_tuple(),
    )
