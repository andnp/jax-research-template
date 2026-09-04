"""Segment-aware metrics for canonical process-control traces."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose

import jax
import jax.numpy as jnp

from process_control.sysid.comparison import (
    TraceComparisonResult,
    TraceComparisonSpec,
    TraceMetricArrays,
)
from process_control.sysid.trace import SignalSpec, UniformTrace


@dataclass(frozen=True, slots=True)
class _MetricSums:
    count: jax.Array
    error_sum: jax.Array
    absolute_error_sum: jax.Array
    squared_error_sum: jax.Array
    maximum_absolute_error: jax.Array
    reference_sum: jax.Array
    candidate_sum: jax.Array
    reference_squared_sum: jax.Array
    candidate_squared_sum: jax.Array
    cross_sum: jax.Array


@dataclass(frozen=True, slots=True)
class _TraceSignal:
    values: jax.Array
    valid: jax.Array
    spec: SignalSpec


def _find_signal(trace: UniformTrace, name: str) -> _TraceSignal:
    groups = (
        (trace.schema.controls, trace.arrays.controls, 0),
        (trace.schema.disturbances, trace.arrays.disturbances, len(trace.schema.controls)),
        (
            trace.schema.measurements,
            trace.arrays.measurements,
            len(trace.schema.controls) + len(trace.schema.disturbances),
        ),
        (
            trace.schema.actuators,
            trace.arrays.actuators,
            len(trace.schema.controls)
            + len(trace.schema.disturbances)
            + len(trace.schema.measurements),
        ),
    )
    for specs, values, offset in groups:
        for index, spec in enumerate(specs):
            if spec.name == name:
                return _TraceSignal(
                    values=values[:, index],
                    valid=trace.arrays.valid[:, offset + index],
                    spec=spec,
                )
    raise ValueError(f"trace does not contain signal {name!r}")


def _metric_sums(
    reference: jax.Array,
    candidate: jax.Array,
    valid: jax.Array,
) -> _MetricSums:
    reference = jnp.asarray(reference, dtype=jnp.float32)
    candidate = jnp.asarray(candidate, dtype=jnp.float32)
    valid = jnp.asarray(valid, dtype=bool)
    error = jnp.where(valid, candidate - reference, 0.0)
    absolute_error = jnp.abs(error)
    safe_reference = jnp.where(valid, reference, 0.0)
    safe_candidate = jnp.where(valid, candidate, 0.0)
    return _MetricSums(
        count=jnp.sum(valid, axis=0, dtype=jnp.int32),
        error_sum=jnp.sum(error, axis=0),
        absolute_error_sum=jnp.sum(absolute_error, axis=0),
        squared_error_sum=jnp.sum(error**2, axis=0),
        maximum_absolute_error=jnp.max(
            jnp.where(valid, absolute_error, 0.0),
            axis=0,
        ),
        reference_sum=jnp.sum(safe_reference, axis=0),
        candidate_sum=jnp.sum(safe_candidate, axis=0),
        reference_squared_sum=jnp.sum(safe_reference**2, axis=0),
        candidate_squared_sum=jnp.sum(safe_candidate**2, axis=0),
        cross_sum=jnp.sum(safe_reference * safe_candidate, axis=0),
    )


def _stack_sums(sums: Sequence[_MetricSums]) -> _MetricSums:
    return _MetricSums(
        count=jnp.stack([value.count for value in sums]),
        error_sum=jnp.stack([value.error_sum for value in sums]),
        absolute_error_sum=jnp.stack([value.absolute_error_sum for value in sums]),
        squared_error_sum=jnp.stack([value.squared_error_sum for value in sums]),
        maximum_absolute_error=jnp.stack(
            [value.maximum_absolute_error for value in sums]
        ),
        reference_sum=jnp.stack([value.reference_sum for value in sums]),
        candidate_sum=jnp.stack([value.candidate_sum for value in sums]),
        reference_squared_sum=jnp.stack(
            [value.reference_squared_sum for value in sums]
        ),
        candidate_squared_sum=jnp.stack(
            [value.candidate_squared_sum for value in sums]
        ),
        cross_sum=jnp.stack([value.cross_sum for value in sums]),
    )


def _summarize(
    sums: _MetricSums,
    scales: jax.Array,
    scale_defined: jax.Array,
) -> TraceMetricArrays:
    count = sums.count.astype(jnp.float32)
    safe_count = jnp.maximum(count, 1.0)
    bias = sums.error_sum / safe_count
    mae = sums.absolute_error_sum / safe_count
    rmse = jnp.sqrt(sums.squared_error_sum / safe_count)
    reference_variance = (
        sums.reference_squared_sum - sums.reference_sum**2 / safe_count
    )
    candidate_variance = (
        sums.candidate_squared_sum - sums.candidate_sum**2 / safe_count
    )
    covariance = (
        sums.cross_sum - sums.reference_sum * sums.candidate_sum / safe_count
    )
    correlation_denominator = jnp.sqrt(
        jnp.maximum(reference_variance, 0.0)
        * jnp.maximum(candidate_variance, 0.0)
    )
    correlation_defined = (sums.count > 1) & (correlation_denominator > 1e-12)
    r2_defined = (sums.count > 1) & (reference_variance > 1e-12)
    correlation = jnp.where(
        correlation_defined,
        covariance / jnp.maximum(correlation_denominator, 1e-12),
        0.0,
    )
    r2 = jnp.where(
        r2_defined,
        1.0 - sums.squared_error_sum / jnp.maximum(reference_variance, 1e-12),
        0.0,
    )
    nrmse_defined = scale_defined & (sums.count > 0)
    nrmse = jnp.where(nrmse_defined, rmse / scales, 0.0)
    return TraceMetricArrays(
        valid_count=sums.count,
        bias=bias,
        mae=mae,
        rmse=rmse,
        maximum_absolute_error=sums.maximum_absolute_error,
        correlation=correlation,
        r2=r2,
        nrmse=nrmse,
        correlation_defined=correlation_defined,
        r2_defined=r2_defined,
        nrmse_defined=nrmse_defined,
    )


def compare_traces(
    reference: UniformTrace,
    candidate: UniformTrace,
    spec: TraceComparisonSpec,
) -> TraceComparisonResult:
    """Compare named signals without crossing invalid or segmented rows."""
    if reference.n_steps != candidate.n_steps:
        raise ValueError("reference and candidate traces must have equal lengths")
    if not isclose(reference.sample_period, candidate.sample_period, rel_tol=0.0, abs_tol=spec.time_tolerance):
        raise ValueError("reference and candidate sample periods must match")
    if not bool(
        jnp.all(
            jnp.abs(reference.time - candidate.time) <= spec.time_tolerance
        )
    ):
        raise ValueError("reference and candidate timestamps must match")
    if spec.warmup_steps >= reference.n_steps:
        raise ValueError("warmup_steps must leave at least one comparison row")

    pairs = tuple(spec.pairs)
    reference_signals = tuple(_find_signal(reference, pair.reference) for pair in pairs)
    candidate_signals = tuple(_find_signal(candidate, pair.candidate) for pair in pairs)
    for pair, reference_signal, candidate_signal in zip(
        pairs,
        reference_signals,
        candidate_signals,
        strict=True,
    ):
        if reference_signal.spec.unit != pair.unit or candidate_signal.spec.unit != pair.unit:
            raise ValueError(f"units do not match comparison pair {pair.reference!r}")

    rows = slice(spec.warmup_steps, None)
    reference_values = jnp.stack([signal.values[rows] for signal in reference_signals], axis=1)
    candidate_values = jnp.stack([signal.values[rows] for signal in candidate_signals], axis=1)
    valid = jnp.stack(
        [reference_signal.valid[rows] & candidate_signal.valid[rows] for reference_signal, candidate_signal in zip(reference_signals, candidate_signals, strict=True)],
        axis=1,
    )
    segment_ids = reference.arrays.segment_id[rows]
    unique_segments = tuple(int(value) for value in jnp.unique(segment_ids).tolist())
    segment_sums = _stack_sums(
        tuple(
            _metric_sums(
                reference_values,
                candidate_values,
                valid & (segment_ids[:, None] == segment_id),
            )
            for segment_id in unique_segments
        )
    )
    scales = jnp.asarray(
        [pair.normalization_scale or 1.0 for pair in pairs],
        dtype=jnp.float32,
    )
    scale_defined = jnp.asarray(
        [pair.normalization_scale is not None for pair in pairs],
        dtype=bool,
    )
    per_segment = _summarize(segment_sums, scales, scale_defined)
    segment_count = jnp.maximum(segment_sums.count.astype(jnp.float32), 1.0)
    within_reference_variance = (
        segment_sums.reference_squared_sum
        - segment_sums.reference_sum**2 / segment_count
    )
    within_candidate_variance = (
        segment_sums.candidate_squared_sum
        - segment_sums.candidate_sum**2 / segment_count
    )
    within_covariance = (
        segment_sums.cross_sum
        - segment_sums.reference_sum * segment_sums.candidate_sum / segment_count
    )
    overall_sums = _MetricSums(
        count=jnp.sum(segment_sums.count, axis=0),
        error_sum=jnp.sum(segment_sums.error_sum, axis=0),
        absolute_error_sum=jnp.sum(segment_sums.absolute_error_sum, axis=0),
        squared_error_sum=jnp.sum(segment_sums.squared_error_sum, axis=0),
        maximum_absolute_error=jnp.max(segment_sums.maximum_absolute_error, axis=0),
        reference_sum=jnp.zeros_like(segment_sums.reference_sum[0]),
        candidate_sum=jnp.zeros_like(segment_sums.candidate_sum[0]),
        reference_squared_sum=jnp.sum(within_reference_variance, axis=0),
        candidate_squared_sum=jnp.sum(within_candidate_variance, axis=0),
        cross_sum=jnp.sum(within_covariance, axis=0),
    )
    return TraceComparisonResult(
        spec=spec,
        segment_ids=jnp.asarray(unique_segments, dtype=jnp.int32),
        per_segment=per_segment,
        overall=_summarize(overall_sums, scales, scale_defined),
    )
