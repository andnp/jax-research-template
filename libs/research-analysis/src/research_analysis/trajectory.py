"""Contracts for validating tidy trajectory frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

import polars as pl


@dataclass(frozen=True)
class TrajectorySchema:
    """Column names and exact dtypes for one tidy trajectory contract."""

    identity: tuple[tuple[str, type[pl.DataType]], ...]
    step: tuple[str, type[pl.DataType]]
    metric: tuple[str, type[pl.DataType]]
    value: tuple[str, type[pl.DataType]]

    @property
    def identity_columns(self) -> tuple[str, ...]:
        """Return required identity column names in contract order."""
        return tuple(column for column, _ in self.identity)

    @property
    def observation_columns(self) -> tuple[str, ...]:
        """Return observation column names in validation order."""
        return (self.step[0], self.metric[0], self.value[0])

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return all required columns in deterministic diagnostic order."""
        return (*self.identity_columns, *self.observation_columns)

    @property
    def column_schema(self) -> tuple[tuple[str, type[pl.DataType]], ...]:
        """Return identity and observation columns with their expected dtypes."""
        return (*self.identity, self.step, self.metric, self.value)


class TrajectoryDiagnosticCode(StrEnum):
    """Validation outcomes for one or more trajectory rows."""

    MISSING_COLUMN = "missing_column"
    INVALID_COLUMN_TYPE = "invalid_column_type"
    EMPTY_METRIC = "empty_metric"
    NON_FINITE_VALUE = "non_finite_value"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    DUPLICATE_STEP = "duplicate_step"
    NON_INCREASING_STEP = "non_increasing_step"
    NON_UNIFORM_GRID = "non_uniform_grid"


@dataclass(frozen=True)
class TrajectoryDiagnostic:
    """Structured validation detail for a trajectory identity."""

    code: TrajectoryDiagnosticCode
    message: str
    identity: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class TrajectoryValidationReport:
    """Validation results for one selected metric."""

    metric: str
    identity_columns: tuple[str, ...]
    n_trajectories: int
    diagnostics: tuple[TrajectoryDiagnostic, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether validation produced no diagnostics."""
        return not self.diagnostics


def trajectory_identity_columns(frame: pl.DataFrame, schema: TrajectorySchema) -> tuple[str, ...]:
    """Return frame columns that identify a trajectory rather than an observation."""
    observation_columns = frozenset(schema.observation_columns)
    return tuple(column for column in frame.columns if column not in observation_columns)


def _identity(group: pl.DataFrame, columns: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
    values = group.row(0)
    positions = {column: group.columns.index(column) for column in columns}
    return tuple((column, values[positions[column]]) for column in columns)


def _diagnostic(
    code: TrajectoryDiagnosticCode,
    message: str,
    identity: tuple[tuple[str, object], ...],
) -> TrajectoryDiagnostic:
    return TrajectoryDiagnostic(code=code, message=message, identity=identity)


def _diagnostic_message(diagnostic: TrajectoryDiagnostic) -> str:
    identity = ", ".join(f"{column}={value!r}" for column, value in diagnostic.identity)
    return f"{identity}: {diagnostic.message}" if identity else diagnostic.message


def validate_trajectories(
    frame: pl.DataFrame,
    metric: str,
    *,
    schema: TrajectorySchema,
) -> TrajectoryValidationReport:
    """Return structured validation results for one metric in a tidy frame."""
    missing_columns = tuple(column for column in schema.required_columns if column not in frame.columns)
    identity_columns = trajectory_identity_columns(frame, schema)
    if missing_columns:
        missing_diagnostics = tuple(
            _diagnostic(
                TrajectoryDiagnosticCode.MISSING_COLUMN,
                f"trajectory frame is missing column {column!r}",
                (),
            )
            for column in missing_columns
        )
        return TrajectoryValidationReport(metric, identity_columns, 0, missing_diagnostics)

    invalid_identity_dtypes = tuple(
        (column, expected)
        for column, expected in schema.identity
        if frame.schema[column] != expected
    )
    if invalid_identity_dtypes:
        dtype_diagnostics = tuple(
            _diagnostic(
                TrajectoryDiagnosticCode.INVALID_COLUMN_TYPE,
                f"trajectory identity column {column!r} must have dtype {expected}",
                (),
            )
            for column, expected in invalid_identity_dtypes
        )
        return TrajectoryValidationReport(metric, identity_columns, 0, dtype_diagnostics)

    data_schema = (schema.step, schema.metric, schema.value)
    invalid_data_dtypes = tuple(
        (column, expected)
        for column, expected in data_schema
        if frame.schema[column] != expected
    )
    if invalid_data_dtypes:
        dtype_diagnostics = tuple(
            _diagnostic(
                TrajectoryDiagnosticCode.INVALID_COLUMN_TYPE,
                f"trajectory data column {column!r} must have dtype {expected}",
                (),
            )
            for column, expected in invalid_data_dtypes
        )
        return TrajectoryValidationReport(metric, identity_columns, 0, dtype_diagnostics)

    selected = frame.filter(pl.col(schema.metric[0]) == metric)
    if selected.is_empty():
        diagnostic = _diagnostic(
            TrajectoryDiagnosticCode.EMPTY_METRIC,
            f"trajectory frame has no rows for metric {metric!r}",
            (),
        )
        return TrajectoryValidationReport(metric, identity_columns, 0, (diagnostic,))

    groups = (
        selected.partition_by(list(identity_columns), maintain_order=True)
        if identity_columns
        else [selected]
    )
    diagnostics: list[TrajectoryDiagnostic] = []
    for group in groups:
        identity = _identity(group, identity_columns)
        steps = group.get_column(schema.step[0]).to_list()
        values = group.get_column(schema.value[0]).to_list()
        if len(steps) < 2:
            diagnostics.append(
                _diagnostic(
                    TrajectoryDiagnosticCode.INSUFFICIENT_OBSERVATIONS,
                    "trajectory must contain at least two observations",
                    identity,
                )
            )
            continue
        if any(not isinstance(step, int) or isinstance(step, bool) for step in steps):
            diagnostics.append(_diagnostic(TrajectoryDiagnosticCode.INVALID_COLUMN_TYPE, "step values must be integers", identity))
            continue
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            diagnostics.append(_diagnostic(TrajectoryDiagnosticCode.NON_FINITE_VALUE, "trajectory values must be finite", identity))
            continue
        if len(set(steps)) != len(steps):
            diagnostics.append(_diagnostic(TrajectoryDiagnosticCode.DUPLICATE_STEP, "trajectory steps must be unique", identity))
            continue
        if any(next_step <= step for step, next_step in pairwise(steps)):
            diagnostics.append(_diagnostic(TrajectoryDiagnosticCode.NON_INCREASING_STEP, "trajectory steps must increase", identity))
            continue
        differences = {next_step - step for step, next_step in pairwise(steps)}
        if len(differences) > 1:
            diagnostics.append(
                _diagnostic(
                    TrajectoryDiagnosticCode.NON_UNIFORM_GRID,
                    "trajectory steps must be uniformly spaced",
                    identity,
                )
            )

    return TrajectoryValidationReport(metric, identity_columns, len(groups), tuple(diagnostics))


def require_valid_trajectories(
    frame: pl.DataFrame,
    metric: str,
    *,
    schema: TrajectorySchema,
) -> TrajectoryValidationReport:
    """Raise ValueError when a selected metric violates the trajectory contract."""
    report = validate_trajectories(frame, metric, schema=schema)
    if not report.is_valid:
        messages = "; ".join(_diagnostic_message(diagnostic) for diagnostic in report.diagnostics)
        raise ValueError(messages)
    return report
