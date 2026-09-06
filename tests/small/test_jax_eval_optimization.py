"""Small behavioral tests for jax-eval optimization diagnostics."""

import jax.numpy as jnp
from jax_eval.optimization import gradient_snr


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
