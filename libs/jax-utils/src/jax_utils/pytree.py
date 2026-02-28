"""Pytree-level math operations.

These helpers operate on arbitrary JAX pytrees, applying element-wise
operations across all leaves and reducing where needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def tree_zeros_like[T](tree: T) -> T:
    """Return a pytree with the same structure, all leaves filled with zeros."""
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_ones_like[T](tree: T) -> T:
    """Return a pytree with the same structure, all leaves filled with ones."""
    return jax.tree_util.tree_map(jnp.ones_like, tree)


def tree_add[T](a: T, b: T) -> T:
    """Element-wise addition of two pytrees with identical structure."""
    return jax.tree_util.tree_map(jnp.add, a, b)


def tree_sub[T](a: T, b: T) -> T:
    """Element-wise subtraction of two pytrees (a - b)."""
    return jax.tree_util.tree_map(jnp.subtract, a, b)


def tree_scalar_mul[T](scalar: float | Array, tree: T) -> T:
    """Multiply every leaf in *tree* by *scalar*."""
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def tree_mean(tree: object) -> Array:
    """Global mean across all leaves of a pytree.

    Computes the mean of all scalar values across all leaves,
    weighting each leaf by its number of elements.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    total_sum = sum(jnp.sum(leaf) for leaf in leaves)
    total_count = sum(leaf.size for leaf in leaves)
    return total_sum / total_count


def tree_std(tree: object) -> Array:
    """Global standard deviation across all leaves of a pytree.

    Uses the population formula: ``sqrt(E[x²] - E[x]²)``.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    total_count = sum(leaf.size for leaf in leaves)
    mean = tree_mean(tree)
    sq_diff_sum = sum(jnp.sum(jnp.square(leaf - mean)) for leaf in leaves)
    return jnp.sqrt(sq_diff_sum / total_count)


def tree_norm(tree: object) -> Array:
    """Global L2 norm across all leaves of a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def tree_inner_product(a: object, b: object) -> Array:
    """Global inner product (dot product) across all leaves of two pytrees."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return sum(jnp.sum(la * lb) for la, lb in zip(leaves_a, leaves_b, strict=True))


def tree_lerp[T](a: T, b: T, t: float | Array) -> T:
    """Linear interpolation between two pytrees: ``a + t * (b - a)``.

    Useful for soft target network updates:
    ``target = tree_lerp(target, online, tau)``.
    """
    return jax.tree_util.tree_map(lambda x, y: x + t * (y - x), a, b)
