"""Small contracts for process-control system-identification traces."""

import jax
import jax.numpy as jnp
import pytest
from process_control.sysid.resampling import (
    ResamplingMethod,
    ResamplingPolicy,
    canonicalize,
    resample_signal,
)
from process_control.sysid.trace import SignalRole, SignalSeries, SignalSpec


def _signal(
    name: str,
    role: SignalRole,
    time: list[float],
    values: list[float],
    valid: list[bool] | None = None,
) -> SignalSeries:
    return SignalSeries(
        time=jnp.asarray(time),
        values=jnp.asarray(values),
        spec=SignalSpec(name, role, "unit"),
        valid=None if valid is None else jnp.asarray(valid),
    )


def test_signal_series_rejects_nonmonotonic_timestamps() -> None:
    """Reject ambiguous native-rate ordering before resampling can occur."""
    with pytest.raises(ValueError, match="strictly increasing"):
        _signal("measurement", SignalRole.MEASUREMENT, [0.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_canonicalize_keeps_role_matrices_and_provenance_aligned() -> None:
    """Keep metadata, values, validity, and counts in the same role order."""
    measurement = _signal("y", SignalRole.MEASUREMENT, [0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
    control = _signal("u", SignalRole.CONTROL, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    trace = canonicalize((measurement, control), jnp.arange(3.0), 1.0)

    assert trace.schema.controls[0].name == "u"
    assert trace.schema.measurements[0].name == "y"
    assert jnp.allclose(trace.arrays.controls[:, 0], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(trace.arrays.measurements[:, 0], jnp.array([10.0, 20.0, 30.0]))
    assert jnp.array_equal(trace.arrays.valid, jnp.ones((3, 2), dtype=bool))
    assert jnp.array_equal(trace.arrays.source_count, jnp.ones((3, 2), dtype=jnp.int32))


def test_resampling_policies_preserve_their_signal_semantics() -> None:
    """Use interval means, bounded interpolation, and exact invalidation explicitly."""
    signal = _signal("y", SignalRole.MEASUREMENT, [0.1, 0.9, 1.1], [1.0, 3.0, 5.0])

    mean_values, mean_valid, mean_count = resample_signal(
        signal,
        jnp.array([0.0, 1.0]),
        1.0,
        ResamplingPolicy(ResamplingMethod.MEAN),
    )
    linear_values, linear_valid, linear_count = resample_signal(
        _signal("linear", SignalRole.MEASUREMENT, [0.0, 1.0], [0.0, 2.0]),
        jnp.array([0.5, 1.5]),
        1.0,
        ResamplingPolicy(ResamplingMethod.LINEAR, max_gap=1.0),
    )
    invalid_values, invalid_valid, invalid_count = resample_signal(
        signal,
        jnp.array([0.0, 1.0, 1.1]),
        1.0,
        ResamplingPolicy(ResamplingMethod.INVALID),
    )

    assert jnp.allclose(mean_values, jnp.array([2.0, 5.0]))
    assert jnp.array_equal(mean_valid, jnp.array([True, True]))
    assert jnp.array_equal(mean_count, jnp.array([2, 1]))
    assert jnp.allclose(linear_values[0], 1.0)
    assert not bool(linear_valid[1])
    assert jnp.array_equal(linear_count, jnp.array([2, 0]))
    assert not bool(invalid_valid[0])
    assert bool(invalid_valid[2])
    assert jnp.isnan(invalid_values[0])
    assert jnp.array_equal(invalid_count, jnp.array([0, 0, 1]))


def test_last_policy_can_bound_stale_values() -> None:
    """Invalidate held values once their permitted source age is exceeded."""
    signal = _signal("u", SignalRole.CONTROL, [0.0, 2.0], [4.0, 8.0])

    values, valid, count = resample_signal(
        signal,
        jnp.array([0.0, 1.0, 2.0]),
        1.0,
        ResamplingPolicy(ResamplingMethod.LAST, max_gap=0.5),
    )

    assert jnp.allclose(values[jnp.array([0, 2])], jnp.array([4.0, 8.0]))
    assert jnp.array_equal(valid, jnp.array([True, False, True]))
    assert jnp.array_equal(count, jnp.array([1, 0, 1]))


def test_gap_segments_remain_stable_until_the_next_valid_run() -> None:
    """Represent one invalid interval as one segment boundary for lagged consumers."""
    signal = _signal(
        "y",
        SignalRole.MEASUREMENT,
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        valid=[True, False, True],
    )

    trace = canonicalize((signal,), jnp.arange(3.0), 1.0)

    assert jnp.array_equal(trace.arrays.valid[:, 0], jnp.array([True, False, True]))
    assert jnp.array_equal(trace.arrays.segment_id, jnp.array([0, 1, 1]))


def test_uniform_trace_payload_is_jit_compatible() -> None:
    """Expose only numeric arrays to compiled process-control functions."""
    signal = _signal("y", SignalRole.MEASUREMENT, [0.0, 1.0], [1.0, 2.0])
    trace = canonicalize((signal,), jnp.arange(2.0), 1.0)

    measurements = jax.jit(lambda payload: payload.measurements)(trace.jax_payload)

    assert jnp.allclose(measurements[:, 0], jnp.array([1.0, 2.0]))
