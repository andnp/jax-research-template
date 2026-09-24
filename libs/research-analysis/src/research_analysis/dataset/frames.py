"""Assemble typed run/metric polars frames from materialized rows."""

from __future__ import annotations

import polars as pl

from research_analysis.dataset.types import ExperimentDimensions, MaterializedRun, SweepAxis, SweepValueType


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


def _provenance_schema_fields():
    return {
        f"{prefix}_{field}": dtype
        for prefix in ("algo", "env")
        for field, dtype in (
            ("component_name", pl.String),
            ("component_type", pl.String),
            ("version_id", pl.Int64),
            ("version_number", pl.Int64),
            ("code_snapshot_hash", pl.String),
        )
    }


def _run_record(run: MaterializedRun) -> dict[str, object]:
    record: dict[str, object] = {
        **run.dimension_values,
        "seed": run.seed,
        "run_id": run.run_id,
        "execution_id": run.execution_id,
        "ablation": run.ablation,
        **{field: None for field in _provenance_schema_fields()},
    }
    for item in run.provenance:
        if item.component_type == "ALGO":
            prefix = "algo"
        elif item.component_type == "ENV":
            prefix = "env"
        else:
            raise ValueError(f"unsupported dataset component type: {item.component_type!r}")
        record.update(
            {
                f"{prefix}_component_name": item.component_name,
                f"{prefix}_component_type": item.component_type,
                f"{prefix}_version_id": item.version_id,
                f"{prefix}_version_number": item.version_number,
                f"{prefix}_code_snapshot_hash": item.code_snapshot_hash,
            }
        )
    return record


def assemble_frames(
    rows: list[MaterializedRun], metric_frames: list[pl.DataFrame], dimensions: ExperimentDimensions
) -> tuple[pl.DataFrame, pl.DataFrame]:
    run_schema = {**_identity_schema_fields(dimensions), **_provenance_schema_fields(), **_axis_schema_fields(dimensions)}
    run_frame = pl.DataFrame([_run_record(row) for row in rows], schema=run_schema)
    metric_frame = (
        pl.concat(metric_frames).sort(["condition_name", "seed", "step", "metric"])
        if metric_frames
        else pl.DataFrame(schema={**_identity_schema_fields(dimensions), "step": pl.Int64, "metric": pl.String, "value": pl.Float64})
    )
    return run_frame, metric_frame


def assemble_scalar_frames(
    rows: list[MaterializedRun], scalar_frames: list[pl.DataFrame], dimensions: ExperimentDimensions
) -> tuple[pl.DataFrame, pl.DataFrame]:
    scalar_schema = {
        **_identity_schema_fields(dimensions),
        **_axis_schema_fields(dimensions),
        "metric": pl.String,
        "value": pl.Float64,
    }
    execution_schema = {"execution_id": pl.Int64, "metric": pl.String, "value": pl.Float64}
    if not scalar_frames:
        return pl.DataFrame(schema=scalar_schema), pl.DataFrame(schema=execution_schema)

    metadata = pl.DataFrame(
        [
            {
                **row.dimension_values,
                "seed": row.seed,
                "run_id": row.run_id,
                "execution_id": row.execution_id,
                "ablation": row.ablation,
            }
            for row in rows
        ],
        schema={**_identity_schema_fields(dimensions), **_axis_schema_fields(dimensions)},
    )
    scalar_rows = pl.concat(scalar_frames).with_columns(pl.col("value").cast(pl.Float64).fill_null(float("nan")))
    run_scalars = (
        scalar_rows.filter(pl.col("scope") == "run")
        .drop(["experiment_id", "seed_id"])
        .join(metadata, on=["run_id", "execution_id"], how="inner")
    )
    execution_metrics = (
        scalar_rows.filter(pl.col("scope") == "execution")
        .select("execution_id", pl.col("metric_name").alias("metric"), "value")
    )
    return (
        run_scalars.rename({"metric_name": "metric"}).select(list(scalar_schema)),
        execution_metrics.select(list(execution_schema)),
    )
