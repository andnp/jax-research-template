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


@dataclass(frozen=True)
class ResolvedDimensions:
    """Validated analysis dimensions for one run."""

    values: Mapping[str, SweepValue | None]


@dataclass(frozen=True)
class _ResolvedAxisValue:
    value: SweepValue | None


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


def _resolve_axis_value(
    axis: SweepAxis, hyperparameters: Mapping[str, ParameterValue]
) -> _ResolvedAxisValue | None:
    if axis.when is not None:
        if any(key not in hyperparameters for key in axis.when):
            return None
        if not all(hyperparameters[key] == value for key, value in axis.when.items()):
            return _ResolvedAxisValue(None)
    if axis.name not in hyperparameters:
        return None
    value = _coerce_axis_value(hyperparameters[axis.name], axis)
    if value is None:
        return None
    return _ResolvedAxisValue(value)


def resolve_dimensions(
    hyperparameters: Mapping[str, ParameterValue],
    dimensions: ExperimentDimensions,
) -> ResolvedDimensions | None:
    source_dimensions = {"condition_name": dimensions.condition}
    if dimensions.environment is not None:
        source_dimensions["env_name"] = dimensions.environment
    values: dict[str, SweepValue | None] = {}
    for output_name, source_name in source_dimensions.items():
        value = hyperparameters.get(source_name)
        if value is None:
            return None
        values[output_name] = str(value)
    for axis in dimensions.sweep_axes:
        axis_value = _resolve_axis_value(axis, hyperparameters)
        if axis_value is None:
            return None
        values[axis.name] = axis_value.value
    return ResolvedDimensions(values)
