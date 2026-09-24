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
class UnresolvedDimension:
    """Why a run's hyperparameters cannot be projected onto the requested dimensions."""

    reason: str


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


def _coerce_dimension_label(value: ParameterValue) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def cohort_matches(
    values: Mapping[str, SweepValue | None], ablation: str | None, selection: CohortSelection
) -> bool:
    if selection.condition_name is not None and values.get("condition_name") != selection.condition_name:
        return False

    if selection.env_name is not None and values.get("env_name") != selection.env_name:
        return False

    if selection.ablation is not None and ablation != selection.ablation:
        return False

    return all(values.get(key) == value for key, value in selection.sweep_values)


def _resolve_axis_value(
    axis: SweepAxis, hyperparameters: Mapping[str, ParameterValue]
) -> _ResolvedAxisValue | UnresolvedDimension:
    if axis.when is not None:
        missing = sorted(key for key in axis.when if key not in hyperparameters)
        if missing:
            return UnresolvedDimension(f"sweep axis {axis.name!r} applicability keys {missing} are missing")

        if not all(hyperparameters[key] in values for key, values in axis.when.items()):
            return _ResolvedAxisValue(None)

    if axis.name not in hyperparameters:
        return UnresolvedDimension(f"declared sweep axis {axis.name!r} is missing")

    value = _coerce_axis_value(hyperparameters[axis.name], axis)
    if value is None:
        return UnresolvedDimension(
            f"declared sweep axis {axis.name!r} expects {axis.value_type} but has {hyperparameters[axis.name]!r}"
        )

    return _ResolvedAxisValue(value)


def _label_mismatch(
    output_name: str, source_name: str, hyperparameters: Mapping[str, ParameterValue]
) -> UnresolvedDimension:
    if source_name in hyperparameters:
        return UnresolvedDimension(
            f"{output_name} dimension key {source_name!r} must be a non-empty string, got {hyperparameters[source_name]!r}"
        )
    string_keys = sorted(key for key, value in hyperparameters.items() if isinstance(value, str))
    return UnresolvedDimension(
        f"{output_name} dimension key {source_name!r} is missing; ExperimentDimensions names the hyperparameter "
        f"holding each run's label (condition defaults to 'condition_name'), so a run labelled by e.g. 'algorithm' "
        f"needs ExperimentDimensions(condition='algorithm'). String hyperparameters on this run: {string_keys}"
    )


def resolve_dimensions(
    hyperparameters: Mapping[str, ParameterValue],
    dimensions: ExperimentDimensions,
) -> ResolvedDimensions | UnresolvedDimension:
    source_dimensions = {"condition_name": dimensions.condition}
    if dimensions.environment is not None:
        source_dimensions["env_name"] = dimensions.environment

    values: dict[str, SweepValue | None] = {}
    for output_name, source_name in source_dimensions.items():
        value = _coerce_dimension_label(hyperparameters.get(source_name))
        if value is None:
            return _label_mismatch(output_name, source_name, hyperparameters)
        values[output_name] = value

    for axis in dimensions.sweep_axes:
        axis_value = _resolve_axis_value(axis, hyperparameters)
        if isinstance(axis_value, UnresolvedDimension):
            return axis_value
        values[axis.name] = axis_value.value

    return ResolvedDimensions(values)
