"""Small behavioral tests for jax-eval optimization diagnostics."""

import jax.numpy as jnp
import pytest
from jax_eval.optimization import gradient_snr, update_ratio


def test_gradient_snr_reduces_nested_leaves_without_leaf_weighting() -> None:
    """Compute the global signal-to-noise ratio over nested leaves.

    Leaf sizes contribute through their elements, without an extra leaf weight.
    """
    grads = {
        "small": jnp.array([[1.0], [3.0]]),
        "nested": {"wide": jnp.array([[0.0, 0.0], [2.0, 2.0]])},
    }

    result = gradient_snr(grads)

    assert jnp.allclose(result, 2.0)


def test_gradient_snr_returns_float32_for_low_precision_inputs() -> None:
    """Upcast gradient leaves before computing the scalar diagnostic."""
    grads = {"weight": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float16)}

    result = gradient_snr(grads)

    assert result.dtype == jnp.float32
    assert jnp.isfinite(result)


def test_gradient_snr_handles_one_identical_zero_batches() -> None:
    """Return a finite large value for one nonzero batch and zero for zero batches."""
    nonzero = gradient_snr({"weight": jnp.array([[[2.0, 0.0]]])})
    identical = gradient_snr({"weight": jnp.array([[2.0, 0.0], [2.0, 0.0]])})
    zero = gradient_snr({"weight": jnp.zeros((2, 2))})

    assert jnp.isfinite(nonzero)
    assert nonzero > 1e6
    assert jnp.isfinite(identical)
    assert identical > 1e6
    assert zero == 0.0


def test_gradient_snr_handles_empty_and_zero_batches() -> None:
    """Return float32 zero for empty pytrees and batches with no entries."""
    empty_result = gradient_snr({})
    zero_batch_result = gradient_snr(
        {"weight": jnp.empty((0, 2)), "nested": {"bias": jnp.empty((0, 1))}}
    )

    assert empty_result.dtype == jnp.float32
    assert empty_result == 0.0
    assert zero_batch_result.dtype == jnp.float32
    assert zero_batch_result == 0.0


def test_gradient_snr_rejects_scalar_and_inconsistent_batch_leaves() -> None:
    """Reject scalar gradients and leaves with different leading batch sizes."""
    with pytest.raises(ValueError):
        gradient_snr({"weight": jnp.array(1.0)})
    with pytest.raises(ValueError):
        gradient_snr({"weight": jnp.ones((2, 1)), "bias": jnp.ones((3, 1))})


def test_update_ratio_uses_global_nested_tree_norms() -> None:
    """Compare update and parameter norms across mixed-size nested leaves."""
    params = {"small": jnp.array([3.0]), "nested": {"wide": jnp.array([4.0, 0.0])}}
    updates = {"small": jnp.array([0.0]), "nested": {"wide": jnp.array([0.0, 3.0])}}

    result = update_ratio(params, updates)

    assert result.dtype == jnp.float32
    assert jnp.allclose(result, 0.6)


def test_update_ratio_handles_zero_params_and_rejects_structure_mismatch() -> None:
    """Return a finite zero-denominator result while rejecting different trees."""
    result = update_ratio({"weight": jnp.zeros(2)}, {"weight": jnp.ones(2)})

    assert jnp.isfinite(result)
    with pytest.raises(ValueError):
        update_ratio({"weight": jnp.ones(2)}, {"bias": jnp.ones(2)})
    with pytest.raises(ValueError):
        update_ratio({"weight": jnp.ones(2)}, {"weight": jnp.ones(3)})
