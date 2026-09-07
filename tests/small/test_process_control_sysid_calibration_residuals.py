"""Small contracts for normalized process-control calibration residuals."""

import jax
import jax.numpy as jnp
import pytest
from process_control.sysid.calibration_residuals import (
    TraceResidualArrays,
    TraceResidualSpec,
    summarize_trace_residuals,
    trace_residuals,
)
from process_control.sysid.comparison import TraceComparisonSpec, TraceSignalPair
from process_control.sysid.resampling import canonicalize
from process_control.sysid.trace import SignalRole, SignalSeries, SignalSpec


def _trace(name: str, values: list[float], valid: list[bool] | None = None):
    time = jnp.arange(len(values), dtype=jnp.float32)
    return canonicalize(
        (
            SignalSeries(
                time=time,
                values=jnp.asarray(values, dtype=jnp.float32),
                spec=SignalSpec(name, SignalRole.MEASUREMENT, "unit"),
                valid=None if valid is None else jnp.asarray(valid),
            ),
        ),
        time,
        1.0,
    )


def _spec(warmup_steps: int = 0) -> TraceResidualSpec:
    return TraceResidualSpec(
        comparison=TraceComparisonSpec(
            pairs=(TraceSignalPair("y", "y", "unit", normalization_scale=2.0),),
            warmup_steps=warmup_steps,
        ),
        weights=(4.0,),
    )


def test_trace_residuals_apply_scale_and_signal_weight() -> None:
    """Normalize each signal before applying its objective weight."""
    reference = _trace("y", [1.0, 3.0, 5.0])
    candidate = _trace("y", [3.0, 1.0, 9.0])

    residuals = trace_residuals(reference, candidate, _spec())
    objective = summarize_trace_residuals(residuals)

    assert residuals.residual.shape == (3, 1)
    assert jnp.allclose(residuals.residual[:, 0], jnp.array([2.0, -2.0, 4.0]))
    assert bool(objective.defined)
    assert int(objective.valid_count) == 3
    assert jnp.allclose(objective.mean_squared_error, jnp.array(8.0))


def test_trace_residuals_preserve_warmup_validity_and_segments() -> None:
    """Keep warmup, invalid rows, and gap identities explicit in the payload."""
    reference = _trace("y", [1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    candidate = _trace("y", [1.0, 20.0, 30.0, 8.0], [True, True, True, True])

    residuals = trace_residuals(reference, candidate, _spec(warmup_steps=1))

    assert residuals.residual.shape == (3, 1)
    assert jnp.array_equal(residuals.valid[:, 0], jnp.array([True, False, True]))
    assert jnp.array_equal(residuals.segment_id, jnp.array([0, 1, 1]))
    assert jnp.allclose(residuals.residual[:, 0], jnp.array([18.0, 0.0, 4.0]))


def test_trace_residual_spec_rejects_mismatched_or_nonpositive_weights() -> None:
    """Reject objective metadata that silently drops or reverses a signal."""
    comparison = TraceComparisonSpec(
        pairs=(TraceSignalPair("y", "y", "unit"),),
    )

    with pytest.raises(ValueError, match="one residual weight"):
        TraceResidualSpec(comparison, weights=())
    with pytest.raises(ValueError, match="finite and positive"):
        TraceResidualSpec(comparison, weights=(0.0,))


def test_empty_residuals_have_stable_undefined_objective() -> None:
    """Represent an unusable case without producing NaN optimizer values."""
    residuals = TraceResidualArrays(
        residual=jnp.ones((2, 1)),
        valid=jnp.zeros((2, 1), dtype=bool),
        segment_id=jnp.array([0, 1], dtype=jnp.int32),
    )

    objective = summarize_trace_residuals(residuals)

    assert int(objective.valid_count) == 0
    assert not bool(objective.defined)
    assert jnp.allclose(objective.mean_squared_error, jnp.array(0.0))


def test_one_fully_invalid_signal_pair_marks_the_objective_undefined() -> None:
    """Never let a divergent signal silently drop out of a multi-signal objective."""
    residuals = TraceResidualArrays(
        residual=jnp.ones((3, 2)),
        valid=jnp.array([[True, False], [True, False], [True, False]]),
        segment_id=jnp.array([0, 0, 0], dtype=jnp.int32),
    )

    objective = summarize_trace_residuals(residuals)

    assert int(objective.valid_count) == 3
    assert not bool(objective.defined)


def test_residual_payload_is_jit_compatible() -> None:
    """Keep residual summaries usable inside compiled calibration code."""
    residuals = TraceResidualArrays(
        residual=jnp.array([[1.0], [3.0]]),
        valid=jnp.ones((2, 1), dtype=bool),
        segment_id=jnp.array([0, 0], dtype=jnp.int32),
    )

    objective = jax.jit(summarize_trace_residuals)(residuals)

    assert int(objective.valid_count) == 2
    assert jnp.allclose(objective.mean_squared_error, jnp.array(5.0))
