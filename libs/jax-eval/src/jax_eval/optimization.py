"""JAX-native optimization diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_utils.pytree import tree_norm, tree_sum_of_squares


def _as_float32(leaf: jax.Array) -> jax.Array:
    return jnp.asarray(leaf, dtype=jnp.float32)


def _mean_batch(leaf: jax.Array) -> jax.Array:
    return jnp.mean(leaf, axis=0, dtype=jnp.float32)


def _subtract_mean(leaf: jax.Array, mean: jax.Array) -> jax.Array:
    return leaf - mean


def _validate_leaf_shape(param: jax.Array, update: jax.Array) -> jax.Array:
    if param.shape != update.shape:
        raise ValueError("params and updates must have matching leaf shapes")
    return param


def gradient_snr(grads_batch: object, *, epsilon: float = 1e-8) -> jax.Array:
    """Return the global signal-to-noise ratio for a batched gradient pytree."""
    grads = jax.tree_util.tree_map(_as_float32, grads_batch)
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        return jnp.float32(0.0)
    if any(leaf.ndim == 0 for leaf in leaves):
        raise ValueError("gradient leaves must have a leading batch dimension")
    batch_size = leaves[0].shape[0]
    if any(leaf.shape[0] != batch_size for leaf in leaves):
        raise ValueError("gradient leaves must share a leading batch size")
    if batch_size == 0:
        return jnp.float32(0.0)

    mean_grads = jax.tree_util.tree_map(_mean_batch, grads)
    centered = jax.tree_util.tree_map(_subtract_mean, grads, mean_grads)
    variance = tree_sum_of_squares(centered) / jnp.float32(batch_size)
    return tree_sum_of_squares(mean_grads) / (variance + epsilon)


def update_ratio(params: object, updates: object, *, epsilon: float = 1e-8) -> jax.Array:
    """Return the global update norm relative to the global parameter norm."""
    params_float = jax.tree_util.tree_map(_as_float32, params)
    updates_float = jax.tree_util.tree_map(_as_float32, updates)
    validated_params = jax.tree_util.tree_map(_validate_leaf_shape, params_float, updates_float)
    return tree_norm(updates_float) / (tree_norm(validated_params) + epsilon)
