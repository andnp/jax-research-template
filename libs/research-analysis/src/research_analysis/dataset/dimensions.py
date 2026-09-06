"""Map one run's hyperparameters to a typed row, with no database dependency."""

from __future__ import annotations

from collections.abc import Mapping

from experiment_definition.records import ExecutionRow, RunRow

from research_analysis.dataset.types import (
    CohortSelection,
    ComponentVersionProvenance,
    ExperimentDimensions,
    SweepAxis,
    SweepValue,
    SweepValueType,
)
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


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


def _axis_applicability(axis: SweepAxis, hyperparameters: Mapping[str, object]) -> bool | None:
    if axis.when is None:
        return True
    for key, value in axis.when.items():
        if key not in hyperparameters:
            return None
        if hyperparameters[key] != value:
            return False
    return True


def cohort_matches(row: dict[str, object], selection: CohortSelection) -> bool:
    if selection.condition_name is not None and row.get("condition_name") != selection.condition_name:
        return False
    if selection.env_name is not None and row.get("env_name") != selection.env_name:
        return False
    if selection.ablation is not None and row.get("ablation") != selection.ablation:
        return False
    return all(row.get(key) == value for key, value in selection.sweep_values)


def build_run_row(
    run: RunRow,
    execution: ExecutionRow,
    metrics_db_path: str,
    items: tuple[ComponentVersionProvenance, ...],
    hyperparameters: Mapping[str, object],
    dimensions: ExperimentDimensions,
    diagnostics: DiagnosticAccumulator,
) -> dict[str, object] | None:
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
    row["metrics_db_path"] = metrics_db_path
    for item in items:
        prefix = "algo" if item.component_type == "ALGO" else "env"
        row[f"{prefix}_component_name"] = item.component_name
        row[f"{prefix}_component_type"] = item.component_type
        row[f"{prefix}_version_id"] = item.version_id
        row[f"{prefix}_version_number"] = item.version_number
        row[f"{prefix}_code_snapshot_hash"] = item.code_snapshot_hash
    return row
