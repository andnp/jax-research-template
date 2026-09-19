"""Contracts for validating tidy trajectory frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
    MISSING_METRIC_ROWS = "missing_metric_rows"
    UNEXPECTED_METRIC_IDENTITY = "unexpected_metric_identity"
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
