"""JAX-native optimization diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_utils.pytree import tree_sum_of_squares


def _as_float32(leaf: jax.Array) -> jax.Array:
    return jnp.asarray(leaf, dtype=jnp.float32)


def _mean_batch(leaf: jax.Array) -> jax.Array:
    return jnp.mean(leaf, axis=0, dtype=jnp.float32)


def _subtract_mean(leaf: jax.Array, mean: jax.Array) -> jax.Array:
    return leaf - mean


def gradient_snr(grads_batch: object, *, epsilon: float = 1e-8) -> jax.Array:
    """Return the global signal-to-noise ratio for a batched gradient pytree."""
    grads = jax.tree_util.tree_map(_as_float32, grads_batch)
    mean_grads = jax.tree_util.tree_map(_mean_batch, grads)
    centered = jax.tree_util.tree_map(_subtract_mean, grads, mean_grads)
    batch_size = jax.tree_util.tree_leaves(grads)[0].shape[0]
    variance = tree_sum_of_squares(centered) / jnp.float32(batch_size)
    return tree_sum_of_squares(mean_grads) / (variance + epsilon)
