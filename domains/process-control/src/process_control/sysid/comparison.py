"""Typed specifications and results for process-control trace comparison."""

from dataclasses import dataclass
from math import isfinite

import jax

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True, slots=True)
class TraceSignalPair:
    """Names and units for one reference-versus-candidate signal comparison."""

    reference: str
    candidate: str
    unit: str
    normalization_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.reference or not self.candidate or not self.unit:
            raise ValueError("signal names and unit must not be empty")
        if self.normalization_scale is not None and (
            not isfinite(self.normalization_scale) or self.normalization_scale <= 0.0
        ):
            raise ValueError("normalization_scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class TraceComparisonSpec:
    """Validation and signal-selection policy for one trace comparison."""

    pairs: tuple[TraceSignalPair, ...]
    warmup_steps: int = 0
    time_tolerance: float = 1e-6
    # False only when the caller otherwise guarantees candidate reuses
    # reference's own time/sample_period verbatim and needs this comparison
    # to run inside a jax.jit trace, where the alignment checks' bool() on a
    # JAX value cannot execute.
    validate_alignment: bool = True

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("at least one signal pair is required")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not isfinite(self.time_tolerance) or self.time_tolerance <= 0.0:
            raise ValueError("time_tolerance must be finite and positive")
        references = tuple(pair.reference for pair in self.pairs)
        candidates = tuple(pair.candidate for pair in self.pairs)
        if len(set(references)) != len(references):
            raise ValueError("reference signal names must be unique")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate signal names must be unique")


@jax_dataclass
class TraceMetricArrays:
    """JAX pytree containing metric columns for signal pairs."""

    valid_count: jax.Array
    bias: jax.Array
    mae: jax.Array
    rmse: jax.Array
    maximum_absolute_error: jax.Array
    correlation: jax.Array
    r2: jax.Array
    nrmse: jax.Array
    correlation_defined: jax.Array
    r2_defined: jax.Array
    nrmse_defined: jax.Array


@dataclass(frozen=True, slots=True)
class TraceComparisonResult:
    """Per-segment and overall metrics for one comparison specification."""

    spec: TraceComparisonSpec
    segment_ids: jax.Array
    per_segment: TraceMetricArrays
    overall: TraceMetricArrays

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Return reference names in metric column order."""
        return tuple(pair.reference for pair in self.spec.pairs)
