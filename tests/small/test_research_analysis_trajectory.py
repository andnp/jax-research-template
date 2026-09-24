"""Small tests for the parameterized tidy trajectory contract."""

from __future__ import annotations

import math

import polars as pl
import pytest
from research_analysis.trajectory import (
    TrajectoryDiagnosticCode,
    TrajectorySchema,
    require_valid_trajectories,
    validate_trajectories,
)

_SCHEMA = TrajectorySchema(
    identity=(("scenario", pl.String), ("replicate", pl.Int64)),
    step=("time", pl.Int64),
    metric=("signal", pl.String),
    value=("measurement", pl.Float64),
)


def _frame(steps: list[int], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scenario": ["baseline"] * len(steps),
            "replicate": [3] * len(steps),
            "batch": ["a"] * len(steps),
            "time": steps,
            "signal": ["reward"] * len(steps),
            "measurement": values,
        }
    )


def test_custom_schema_preserves_required_and_extra_identity_columns() -> None:
    """
    Verify renamed observation columns and extra dimensions remain usable identities.
    """
    report = validate_trajectories(_frame([0, 10, 20], [1.0, 2.0, 3.0]), "reward", schema=_SCHEMA)

    assert report.is_valid
    assert report.identity_columns == ("scenario", "replicate", "batch")
    assert report.n_trajectories == 1


def test_empty_identity_schema_validates_one_trajectory() -> None:
    """
    Verify a frame without identity dimensions is treated as one trajectory.
    """
    schema = TrajectorySchema(identity=(), step=("time", pl.Int64), metric=("signal", pl.String), value=("measurement", pl.Float64))
    frame = pl.DataFrame({"time": [0, 10], "signal": ["reward", "reward"], "measurement": [1.0, 2.0]})

    report = validate_trajectories(frame, "reward", schema=schema)

    assert report.is_valid
    assert report.identity_columns == ()
    assert report.n_trajectories == 1


def test_missing_columns_follow_schema_order() -> None:
    """
    Verify missing-column diagnostics are deterministic across identity and data roles.
    """
    frame = _frame([0, 10], [1.0, 2.0]).drop(["scenario", "time", "measurement"])

    report = validate_trajectories(frame, "reward", schema=_SCHEMA)

    assert [diagnostic.message for diagnostic in report.diagnostics] == [
        "trajectory frame is missing column 'scenario'",
        "trajectory frame is missing column 'time'",
        "trajectory frame is missing column 'measurement'",
    ]


@pytest.mark.parametrize(
    ("column", "dtype"),
    [("scenario", pl.Categorical), ("time", pl.Int32), ("signal", pl.Categorical), ("measurement", pl.Float32)],
)
def test_required_columns_use_exact_dtypes(column: str, dtype: pl.DataType) -> None:
    """
    Verify identity and observation columns reject non-canonical dtypes.
    """
    frame = _frame([0, 10, 20], [1.0, 2.0, 3.0]).with_columns(pl.col(column).cast(dtype))

    report = validate_trajectories(frame, "reward", schema=_SCHEMA)

    assert report.diagnostics[0].code is TrajectoryDiagnosticCode.INVALID_COLUMN_TYPE
    assert column in report.diagnostics[0].message


@pytest.mark.parametrize(
    ("frame", "code"),
    [
        (_frame([0, 10, 20], [1.0, math.nan, 3.0]), TrajectoryDiagnosticCode.NON_FINITE_VALUE),
        (_frame([0, 10, 10], [1.0, 2.0, 3.0]), TrajectoryDiagnosticCode.DUPLICATE_STEP),
        (_frame([0, 20, 10], [1.0, 2.0, 3.0]), TrajectoryDiagnosticCode.NON_INCREASING_STEP),
        (_frame([0, 10, 30], [1.0, 2.0, 3.0]), TrajectoryDiagnosticCode.NON_UNIFORM_GRID),
    ],
)
def test_invalid_trajectory_reports_specific_failure(frame: pl.DataFrame, code: TrajectoryDiagnosticCode) -> None:
    """
    Verify malformed observations produce their specific structured diagnostic.
    """
    report = validate_trajectories(frame, "reward", schema=_SCHEMA)

    assert [diagnostic.code for diagnostic in report.diagnostics] == [code]
    assert dict(report.diagnostics[0].identity)["scenario"] == "baseline"


def test_empty_metric_reports_metric_name() -> None:
    """
    Verify selecting an absent metric returns an empty-metric diagnostic.
    """
    frame = _frame([0, 10], [1.0, 2.0]).with_columns(pl.lit("other").alias("signal"))

    report = validate_trajectories(frame, "reward", schema=_SCHEMA)

    assert report.diagnostics[0].code is TrajectoryDiagnosticCode.EMPTY_METRIC
    assert "'reward'" in report.diagnostics[0].message


def test_short_trajectory_reports_insufficient_observations() -> None:
    """
    Verify a single observation is rejected before grid checks.
    """
    report = validate_trajectories(_frame([0], [1.0]), "reward", schema=_SCHEMA)

    assert report.diagnostics[0].code is TrajectoryDiagnosticCode.INSUFFICIENT_OBSERVATIONS


def test_validation_checks_multiple_identities() -> None:
    """
    Verify each identity is checked independently and reported by its values.
    """
    invalid = _frame([0, 10, 30], [4.0, 5.0, 6.0]).with_columns(
        pl.lit("candidate").alias("scenario"),
        pl.lit(4, dtype=pl.Int64).alias("replicate"),
    )

    report = validate_trajectories(pl.concat([_frame([0, 10, 20], [1.0, 2.0, 3.0]), invalid]), "reward", schema=_SCHEMA)

    assert report.n_trajectories == 2
    assert [diagnostic.code for diagnostic in report.diagnostics] == [TrajectoryDiagnosticCode.NON_UNIFORM_GRID]
    assert dict(report.diagnostics[0].identity)["replicate"] == 4


def test_strict_validation_includes_identity_in_value_error() -> None:
    """
    Verify strict validation preserves the diagnostic identity in its failure text.
    """
    with pytest.raises(ValueError, match="unique") as error:
        require_valid_trajectories(_frame([0, 10, 10], [1.0, 2.0, 3.0]), "reward", schema=_SCHEMA)

    assert "scenario='baseline'" in str(error.value)
    assert "replicate=3" in str(error.value)
    assert "batch='a'" in str(error.value)
