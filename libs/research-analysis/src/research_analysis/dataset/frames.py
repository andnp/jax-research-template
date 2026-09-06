"""Assemble typed run/metric polars frames from materialized rows."""

from __future__ import annotations

import polars as pl

from research_analysis.dataset.types import ExperimentDimensions, SweepAxis, SweepValueType


def _axis_dtype(axis: SweepAxis):
    if axis.value_type is SweepValueType.STRING:
        return pl.String
    if axis.value_type is SweepValueType.INTEGER:
        return pl.Int64
    if axis.value_type is SweepValueType.FLOAT:
        return pl.Float64
    return pl.Boolean


def _identity_schema_fields(dimensions: ExperimentDimensions):
    return {
        "condition_name": pl.String,
        **({"env_name": pl.String} if dimensions.environment is not None else {}),
        "seed": pl.Int64,
        "run_id": pl.Int64,
        "execution_id": pl.Int64,
        "ablation": pl.String,
    }


def _axis_schema_fields(dimensions: ExperimentDimensions):
    return {axis.name: _axis_dtype(axis) for axis in dimensions.sweep_axes}


def assemble_frames(
    rows: list[dict[str, object]], metric_frames: list[pl.DataFrame], dimensions: ExperimentDimensions
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if rows:
        run_frame = pl.DataFrame(rows, schema_overrides=_axis_schema_fields(dimensions))
    else:
        run_frame = pl.DataFrame(
            schema={**_identity_schema_fields(dimensions), "metrics_db_path": pl.String, **_axis_schema_fields(dimensions)}
        )
    metric_frame = (
        pl.concat(metric_frames).sort(["condition_name", "seed", "step", "metric"])
        if metric_frames
        else pl.DataFrame(schema={**_identity_schema_fields(dimensions), "step": pl.Int64, "metric": pl.String, "value": pl.Float64})
    )
    return run_frame, metric_frame
