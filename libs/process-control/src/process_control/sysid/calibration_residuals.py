"""Normalized trace residuals for simulator calibration."""

from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.sysid.comparison import TraceComparisonSpec
from process_control.sysid.comparison_arrays import prepare_trace_comparison
from process_control.sysid.trace import UniformTrace


@dataclass(frozen=True, slots=True)
class TraceResidualSpec:
    """Signal weights layered on top of a named trace comparison."""

    comparison: TraceComparisonSpec
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.comparison.pairs):
            raise ValueError("one residual weight is required per signal pair")
        if not all(isfinite(weight) and weight > 0.0 for weight in self.weights):
            raise ValueError("residual weights must be finite and positive")


@jax_dataclass
class TraceResidualArrays:
    """JAX pytree containing normalized residual rows and validity."""

    residual: jax.Array
    valid: jax.Array
    segment_id: jax.Array


@jax_dataclass
class TraceObjectiveArrays:
    """JAX pytree containing the scalar objective summary."""

    valid_count: jax.Array
    mean_squared_error: jax.Array
    defined: jax.Array


def trace_residuals(
    reference: UniformTrace,
    candidate: UniformTrace,
    spec: TraceResidualSpec,
) -> TraceResidualArrays:
    """Build weighted normalized residuals without crossing invalid samples."""
    aligned = prepare_trace_comparison(reference, candidate, spec.comparison)
    scales = jnp.asarray(
        [pair.normalization_scale or 1.0 for pair in spec.comparison.pairs],
        dtype=jnp.float32,
    )
    weights = jnp.asarray(spec.weights, dtype=jnp.float32)
    safe_reference = jnp.where(aligned.valid, aligned.reference, 0.0)
    safe_candidate = jnp.where(aligned.valid, aligned.candidate, 0.0)
    residual = (safe_candidate - safe_reference) / scales
    return TraceResidualArrays(
        residual=residual * jnp.sqrt(weights),
        valid=aligned.valid,
        segment_id=aligned.segment_id,
    )


def summarize_trace_residuals(residuals: TraceResidualArrays) -> TraceObjectiveArrays:
    """Summarize valid residuals as a stable scalar calibration objective."""
    valid_count = jnp.sum(residuals.valid, dtype=jnp.int32)
    safe_count = jnp.maximum(valid_count, 1).astype(jnp.float32)
    squared = jnp.where(residuals.valid, residuals.residual**2, 0.0)
    return TraceObjectiveArrays(
        valid_count=valid_count,
        mean_squared_error=jnp.sum(squared) / safe_count,
        defined=valid_count > 0,
    )
