"""Map one run's hyperparameters to a typed row, with no database dependency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from experiment_definition.parameter import ParameterValue

from research_analysis.dataset.types import (
    CohortSelection,
    ExperimentDimensions,
    SweepAxis,
    SweepValue,
    SweepValueType,
)
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


@dataclass(frozen=True)
class ResolvedDimensions:
    """Validated analysis dimensions for one run."""

    condition_name: str
    environment_name: str | None
    sweep_values: Mapping[str, SweepValue | None]


def _coerce_axis_value(value: ParameterValue, axis: SweepAxis) -> SweepValue | None:
    if axis.value_type is SweepValueType.STRING and isinstance(value, str):
        return value
    if axis.value_type is SweepValueType.INTEGER and isinstance(value, int) and not isinstance(value, bool):
        return value
    if axis.value_type is SweepValueType.FLOAT and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if axis.value_type is SweepValueType.BOOLEAN and isinstance(value, bool):
        return value
    return None


def cohort_matches(row: Mapping[str, object], selection: CohortSelection) -> bool:
    if selection.condition_name is not None and row.get("condition_name") != selection.condition_name:
        return False
    if selection.env_name is not None and row.get("env_name") != selection.env_name:
        return False
    if selection.ablation is not None and row.get("ablation") != selection.ablation:
        return False
    return all(row.get(key) == value for key, value in selection.sweep_values)


def resolve_dimensions(
    hyperparameters: Mapping[str, ParameterValue],
    dimensions: ExperimentDimensions,
    run_id: int,
    execution_id: int,
    diagnostics: DiagnosticAccumulator,
) -> ResolvedDimensions | None:
    condition_value = hyperparameters.get(dimensions.condition)
    if condition_value is None:
        diagnostics.add(
            DiagnosticCode.MISSING_DIMENSION,
            f"missing condition key {dimensions.condition!r}",
            run_id,
            execution_id,
        )
        return None
    environment_name = None
    if dimensions.environment is not None:
        environment_value = hyperparameters.get(dimensions.environment)
        if environment_value is None:
            diagnostics.add(
                DiagnosticCode.MISSING_DIMENSION,
                f"missing environment key {dimensions.environment!r}",
                run_id,
                execution_id,
            )
            return None
        environment_name = str(environment_value)
    sweep_values: dict[str, SweepValue | None] = {}
    for axis in dimensions.sweep_axes:
        if axis.when is not None and any(key not in hyperparameters for key in axis.when):
            diagnostics.add(
                DiagnosticCode.INVALID_DIMENSION,
                f"cannot evaluate applicability for sweep axis {axis.name!r}",
                run_id,
                execution_id,
            )
            return None
        applies = axis.when is None or all(hyperparameters[key] == value for key, value in axis.when.items())
        if not applies:
            if axis.name in hyperparameters:
                diagnostics.add(
                    DiagnosticCode.INVALID_DIMENSION,
                    f"sweep axis {axis.name!r} is present outside its applicability condition",
                    run_id,
                    execution_id,
                )
                return None
            sweep_values[axis.name] = None
            continue
        if axis.name not in hyperparameters:
            diagnostics.add(
                DiagnosticCode.MISSING_DIMENSION,
                f"missing sweep axis {axis.name!r} for an applicable run",
                run_id,
                execution_id,
            )
            return None
        raw_value = hyperparameters[axis.name]
        value = _coerce_axis_value(raw_value, axis)
        if value is None:
            diagnostics.add(
                DiagnosticCode.INVALID_DIMENSION,
                f"invalid value for sweep axis {axis.name!r}",
                run_id,
                execution_id,
            )
            return None
        sweep_values[axis.name] = value
    return ResolvedDimensions(str(condition_value), environment_name, sweep_values)
