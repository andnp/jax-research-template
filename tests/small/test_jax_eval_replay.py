"""Small behavioral tests for jax-eval replay diagnostics."""

import warnings

import jax.numpy as jnp
import pytest
from jax_eval.replay import priority_utilization, sample_staleness


def test_priority_utilization_normalizes_entropy() -> None:
    """Return one for uniform priorities and normalized entropy otherwise."""
    uniform = priority_utilization(jnp.ones(4))
    uneven = priority_utilization(jnp.array([1.0, 3.0]))

    expected = -(0.25 * jnp.log(0.25) + 0.75 * jnp.log(0.75)) / jnp.log(2.0)
    assert jnp.allclose(uniform, 1.0)
    assert jnp.allclose(uneven, expected)


def test_priority_utilization_ignores_zero_probability_entries() -> None:
    """Exclude zero entries from entropy while retaining their capacity slots."""
    result = priority_utilization(jnp.array([1.0, 0.0, 1.0, 0.0]))

    assert jnp.allclose(result, 0.5)


def test_priority_utilization_rejects_non_vector_inputs() -> None:
    """Reject scalar and batched priorities instead of inferring an axis."""
    with pytest.raises(ValueError):
        priority_utilization(jnp.array(1.0))
    with pytest.raises(ValueError):
        priority_utilization(jnp.ones((2, 2)))


def test_priority_utilization_handles_empty_and_concentrated_inputs() -> None:
    """Return finite float32 zero for degenerate and concentrated distributions."""
    empty = priority_utilization(jnp.empty((0,)))
    capacity_one = priority_utilization(jnp.ones((1,)))
    concentrated = priority_utilization(jnp.array([1.0, 0.0, 0.0, 0.0]))
    zero = priority_utilization(jnp.zeros((4,)))

    assert empty.dtype == jnp.float32
    assert empty == 0.0
    assert capacity_one == 0.0
    assert concentrated == 0.0
    assert jnp.isfinite(zero)
    assert zero == 0.0


def test_sample_staleness_casts_before_signed_subtraction() -> None:
    """Compute small signed ages for large timestamps without underflow or clipping."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        current = jnp.uint32(2**24 + 1)
        result = sample_staleness(current, jnp.array([2**24], dtype=jnp.uint32))
        unsigned_negative = sample_staleness(jnp.uint32(2), jnp.array([3], dtype=jnp.uint32))
        signed_negative = sample_staleness(jnp.int32(2), jnp.array([3], dtype=jnp.int32))
        mixed_positive = sample_staleness(jnp.int32(2**24 + 1), jnp.array([2**24], dtype=jnp.uint32))
        mixed_negative = sample_staleness(jnp.uint32(2**24), jnp.array([2**24 + 1], dtype=jnp.int32))
        opposite_positive = sample_staleness(jnp.uint32(2**32 - 1), jnp.array([-1], dtype=jnp.int32))
        opposite_negative = sample_staleness(jnp.int32(-1), jnp.array([2**32 - 1], dtype=jnp.uint32))

    assert result.dtype == jnp.float32
    assert jnp.allclose(result, 1.0)
    assert jnp.allclose(unsigned_negative, -1.0)
    assert jnp.allclose(signed_negative, -1.0)
    assert jnp.allclose(mixed_positive, 1.0)
    assert jnp.allclose(mixed_negative, -1.0)
    assert jnp.allclose(opposite_positive, jnp.float32(2**32))
    assert jnp.allclose(opposite_negative, jnp.float32(-2**32))
    assert not any("truncat" in str(w.message).lower() for w in caught)


def test_sample_staleness_applies_same_shape_mask() -> None:
    """Average only selected ages while preserving negative values."""
    sampled = jnp.array([[7, 11], [8, 20]], dtype=jnp.int32)
    mask = jnp.array([[True, False], [False, True]])

    result = sample_staleness(jnp.int32(10), sampled, mask)

    assert jnp.allclose(result, -3.5)


def test_sample_staleness_handles_empty_and_no_valid_samples() -> None:
    """Return finite float32 zero when no sample contributes to the mean."""
    empty = sample_staleness(jnp.float32(10.0), jnp.empty((0, 2), dtype=jnp.int32))
    none_valid = sample_staleness(jnp.float32(10.0), jnp.array([7, 8]), jnp.array([False, False]))

    assert empty.dtype == jnp.float32
    assert empty == 0.0
    assert none_valid.dtype == jnp.float32
    assert none_valid == 0.0


def test_sample_staleness_rejects_mask_shape_mismatch() -> None:
    """Reject masks that cannot identify each sampled transition."""
    with pytest.raises(ValueError):
        sample_staleness(jnp.int32(10), jnp.ones((2, 2)), jnp.ones((2, 1), dtype=bool))
