"""Small behavioral tests for jax-eval representation diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax_eval.representations import dead_neurons, feature_drift, linear_probe_r2


def test_dead_neurons_counts_units_across_observations() -> None:
    """Count dead units rather than zero-valued entries.

    A single activation keeps its unit alive even when other observations are zero.
    """
    activations = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    result = dead_neurons(activations)

    assert jnp.allclose(result, 2.0 / 3.0)
    assert jnp.allclose(dead_neurons(jnp.array([0.0, 2.0])), 0.5)


def test_dead_neurons_handles_empty_inputs_and_tolerance() -> None:
    """Return finite zero values for empty inputs and honor the threshold."""
    assert jnp.allclose(dead_neurons(jnp.empty((0, 3))), 0.0)
    assert jnp.allclose(dead_neurons(jnp.empty((4, 0))), 0.0)
    assert jnp.allclose(dead_neurons(jnp.array([[0.01, 0.2], [-0.01, 0.3]]), atol=0.02), 0.5)


def test_representation_kernels_upcast_inputs_and_reduce_multidimensional_batches() -> None:
    """Return float32 results for low-precision inputs across all observation axes."""
    activations = jnp.zeros((2, 2, 3), dtype=jnp.float16).at[..., 1].set(1.0)
    old = jnp.ones((2, 2, 3), dtype=jnp.bfloat16)
    new = old.at[0, 0].set(jnp.array([1.0, -1.0, 1.0], dtype=jnp.bfloat16))

    dead_result = dead_neurons(activations)
    drift_result = feature_drift(old, new)

    assert dead_result.dtype == jnp.float32
    assert drift_result.dtype == jnp.float32
    assert jnp.allclose(dead_result, 2.0 / 3.0)
    assert jnp.allclose(drift_result, (1.0 - 1.0 / 3.0) / 4.0)


def test_feature_drift_computes_clipped_cosine_distance() -> None:
    """Compute row-wise distance with zero-norm safeguards and clipping."""
    old = jnp.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    new = jnp.array([[1.0, 0.0], [2.0, 0.0], [1.0, -1.0]])

    result = feature_drift(old, new)

    assert jnp.allclose(result, (0.0 + 1.0 + 1.0) / 3.0)
    assert jnp.allclose(feature_drift(jnp.array([[1.0, 1e-9]]), jnp.array([[1.0, 0.0]])), 0.0)


def test_feature_drift_handles_empty_observations_and_shape_mismatch() -> None:
    """Return zero for empty observations while rejecting incompatible shapes."""
    assert jnp.allclose(feature_drift(jnp.empty((0, 3)), jnp.empty((0, 3))), 0.0)

    with pytest.raises(ValueError):
        feature_drift(jnp.ones((2, 3)), jnp.ones((2, 4)))
    with pytest.raises(ValueError):
        feature_drift(jnp.ones((2, 1, 3)), jnp.ones((2, 3)))


def test_linear_probe_r2_fits_intercepts_and_multiple_targets() -> None:
    """Fit centered multi-target linear relationships without clipping the score."""
    features = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    targets = jnp.stack((2.0 * features[:, 0] + 3.0, -features[:, 1] + 1.0), axis=1)

    result = linear_probe_r2(features, targets, ridge=1e-8)

    assert jnp.allclose(result, 1.0, atol=1e-5)


def test_linear_probe_r2_preserves_negative_scores() -> None:
    """Expose a probe that predicts worse than the target mean as negative R²."""
    features = jnp.array([[1.0], [2.0], [3.0]])
    targets = jnp.array([[10.0], [11.0], [10.0]])

    result = linear_probe_r2(features, targets, fit_intercept=False)

    assert bool(result < 0.0)
