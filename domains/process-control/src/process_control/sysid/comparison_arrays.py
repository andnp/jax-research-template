"""Shared aligned arrays for process-control trace consumers."""

from dataclasses import dataclass
from math import isclose

import jax
import jax.numpy as jnp

from process_control.sysid.comparison import TraceComparisonSpec
from process_control.sysid.trace import SignalSpec, UniformTrace


@dataclass(frozen=True, slots=True)
class TraceComparisonArrays:
    """Aligned signal arrays selected by a comparison specification."""

    reference: jax.Array
    candidate: jax.Array
    valid: jax.Array
    segment_id: jax.Array


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


def prepare_trace_comparison(
    reference: UniformTrace,
    candidate: UniformTrace,
    spec: TraceComparisonSpec,
) -> TraceComparisonArrays:
    """Validate and align named trace signals for downstream consumers."""
    if reference.n_steps != candidate.n_steps:
        raise ValueError("reference and candidate traces must have equal lengths")
    if spec.warmup_steps >= reference.n_steps:
        raise ValueError("warmup_steps must leave at least one comparison row")
    if spec.validate_alignment:
        if not isclose(
            reference.sample_period,
            candidate.sample_period,
            rel_tol=0.0,
            abs_tol=spec.time_tolerance,
        ):
            raise ValueError("reference and candidate sample periods must match")
        if not bool(jnp.all(jnp.abs(reference.time - candidate.time) <= spec.time_tolerance)):
            raise ValueError("reference and candidate timestamps must match")

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
    reference_values = jnp.stack(
        [signal.values[rows] for signal in reference_signals],
        axis=1,
    )
    candidate_values = jnp.stack(
        [signal.values[rows] for signal in candidate_signals],
        axis=1,
    )
    valid = jnp.stack(
        [
            signal.valid[rows]
            & jnp.isfinite(signal.values[rows])
            for signal in reference_signals
        ],
        axis=1,
    ) & jnp.stack(
        [
            signal.valid[rows]
            & jnp.isfinite(signal.values[rows])
            for signal in candidate_signals
        ],
        axis=1,
    )
    return TraceComparisonArrays(
        reference=reference_values,
        candidate=candidate_values,
        valid=valid,
        segment_id=reference.arrays.segment_id[rows],
    )
