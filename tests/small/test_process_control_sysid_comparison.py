"""Small contracts for segmented process-control trace comparison."""

import jax
import jax.numpy as jnp
import pytest
from process_control.sysid.comparison import TraceComparisonSpec, TraceSignalPair
from process_control.sysid.comparison_metrics import TraceMetricArrays, compare_traces
from process_control.sysid.resampling import canonicalize
from process_control.sysid.trace import SignalRole, SignalSeries, SignalSpec


def _trace(
    name: str,
    values: list[float],
    valid: list[bool] | None = None,
    time_offset: float = 0.0,
    unit: str = "unit",
):
    time = jnp.arange(len(values), dtype=jnp.float32) + time_offset
    return canonicalize(
        (
            SignalSeries(
                time=time,
                values=jnp.asarray(values, dtype=jnp.float32),
                spec=SignalSpec(name, SignalRole.MEASUREMENT, unit),
                valid=None if valid is None else jnp.asarray(valid),
            ),
        ),
        time,
        1.0,
    )


def _spec(
    candidate: str = "y",
    warmup_steps: int = 0,
    normalization_scale: float | None = None,
) -> TraceComparisonSpec:
    return TraceComparisonSpec(
        pairs=(TraceSignalPair("y", candidate, "unit", normalization_scale),),
        warmup_steps=warmup_steps,
    )


def test_exact_matching_traces_have_zero_error() -> None:
    """Report zero error for identical valid trajectories."""
    trace = _trace("y", [1.0, 2.0, 3.0])

    result = compare_traces(trace, trace, _spec(normalization_scale=2.0))

    assert jnp.array_equal(result.segment_ids, jnp.array([0]))
    assert jnp.array_equal(result.overall.valid_count, jnp.array([3]))
    assert jnp.allclose(result.overall.rmse, jnp.array([0.0]))
    assert jnp.allclose(result.overall.bias, jnp.array([0.0]))
    assert bool(result.overall.correlation_defined[0])
    assert bool(result.overall.r2_defined[0])
    assert bool(result.overall.nrmse_defined[0])


def test_named_mapping_and_gap_boundaries_are_preserved() -> None:
    """Compare renamed signals without allowing a gap to join segments."""
    valid = [True, True, False, True, True]
    reference = _trace("y", [1.0, 2.0, 0.0, 10.0, 11.0], valid)
    candidate = _trace("sim_y", [2.0, 4.0, 0.0, 13.0, 9.0], valid)

    result = compare_traces(reference, candidate, _spec("sim_y"))

    assert jnp.array_equal(result.segment_ids, jnp.array([0, 1]))
    assert jnp.array_equal(result.per_segment.valid_count, jnp.array([[2], [2]]))
    assert jnp.allclose(result.per_segment.bias[:, 0], jnp.array([1.5, 0.5]))
    assert jnp.allclose(result.per_segment.correlation[:, 0], jnp.array([1.0, -1.0]))
    assert jnp.allclose(result.overall.rmse, jnp.array([jnp.sqrt(4.5)]))
    assert jnp.allclose(result.overall.correlation, jnp.array([-1.0 / jnp.sqrt(10.0)]))


def test_warmup_rows_are_excluded_from_metric_counts() -> None:
    """Exclude initialization rows while retaining later segment metrics."""
    reference = _trace("y", [1.0, 2.0, 3.0])
    candidate = _trace("y", [100.0, 2.0, 4.0])

    result = compare_traces(reference, candidate, _spec(warmup_steps=1))

    assert jnp.array_equal(result.overall.valid_count, jnp.array([2]))
    assert jnp.allclose(result.overall.bias, jnp.array([0.5]))


def test_constant_signals_mark_correlation_and_r2_undefined() -> None:
    """Return ordinary errors while flagging variance-dependent metrics."""
    reference = _trace("y", [2.0, 2.0, 2.0])
    candidate = _trace("y", [3.0, 3.0, 3.0])

    result = compare_traces(reference, candidate, _spec())

    assert jnp.allclose(result.overall.mae, jnp.array([1.0]))
    assert not bool(result.overall.correlation_defined[0])
    assert not bool(result.overall.r2_defined[0])
    assert not bool(result.overall.nrmse_defined[0])
    assert jnp.allclose(result.overall.correlation, jnp.array([0.0]))
    assert jnp.allclose(result.overall.r2, jnp.array([0.0]))


def test_comparison_requires_matching_grid_and_units() -> None:
    """Reject comparisons that would require hidden alignment or conversion."""
    reference = _trace("y", [1.0, 2.0, 3.0])
    shifted = _trace("y", [1.0, 2.0, 3.0], time_offset=0.5)
    wrong_units = _trace("y", [1.0, 2.0, 3.0], unit="other")

    with pytest.raises(ValueError, match="timestamps must match"):
        compare_traces(reference, shifted, _spec())
    with pytest.raises(ValueError, match="units do not match"):
        compare_traces(reference, wrong_units, _spec())


def test_metric_payload_is_jit_compatible() -> None:
    """Keep numeric comparison outputs usable by compiled consumers."""
    reference = _trace("y", [1.0, 2.0, 3.0])
    candidate = _trace("y", [2.0, 2.0, 4.0])
    result = compare_traces(reference, candidate, _spec())

    def rmse_of(metrics: TraceMetricArrays):
        return metrics.rmse

    rmse = jax.jit(rmse_of)(result.overall)

    assert jnp.allclose(rmse, jnp.array([jnp.sqrt(2.0 / 3.0)]))
