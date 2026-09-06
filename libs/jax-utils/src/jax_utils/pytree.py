"""Pytree-level math operations.

These helpers operate on arbitrary JAX pytrees, applying element-wise
operations across all leaves and reducing where needed.
"""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
from flax.typing import VariableDict
from jax import Array


def _sum_arrays(values: Iterable[Array]) -> Array:
    """Accumulate scalar JAX arrays without introducing Python scalar unions."""
    total = jnp.asarray(0.0)
    for value in values:
        total = total + value
    return total


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

    def scale(x: Array) -> Array:
        return scalar * x

    return jax.tree_util.tree_map(scale, tree)


def tree_mean(tree: object) -> Array:
    """Global mean across all leaves of a pytree.

    Computes the mean of all scalar values across all leaves,
    weighting each leaf by its number of elements.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    total_sum = _sum_arrays(jnp.sum(leaf) for leaf in leaves)
    total_count = sum(leaf.size for leaf in leaves)
    return total_sum / total_count


def tree_std(tree: object) -> Array:
    """Global standard deviation across all leaves of a pytree.

    Uses the population formula: ``sqrt(E[x²] - E[x]²)``.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    total_count = sum(leaf.size for leaf in leaves)
    mean = tree_mean(tree)
    sq_diff_sum = _sum_arrays(jnp.sum(jnp.square(leaf - mean)) for leaf in leaves)
    return jnp.sqrt(sq_diff_sum / total_count)


def tree_sum_of_squares(tree: object) -> Array:
    """Sum of squared values across all leaves of a pytree (an L2 regularization penalty)."""
    leaves = jax.tree_util.tree_leaves(tree)
    return _sum_arrays(jnp.sum(jnp.square(leaf)) for leaf in leaves)


def tree_norm(tree: object) -> Array:
    """Global L2 norm across all leaves of a pytree."""
    return jnp.sqrt(tree_sum_of_squares(tree))


def tree_inner_product(a: object, b: object) -> Array:
    """Global inner product (dot product) across all leaves of two pytrees."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return _sum_arrays(jnp.sum(la * lb) for la, lb in zip(leaves_a, leaves_b, strict=True))


def tree_lerp[T](a: T, b: T, t: float | Array) -> T:
    """Linear interpolation between two pytrees: ``a + t * (b - a)``.

    Useful for soft target network updates:
    ``target = tree_lerp(target, online, tau)``.
    """
    def lerp(x: Array, y: Array) -> Array:
        return x + t * (y - x)

    return jax.tree_util.tree_map(lerp, a, b)


def shift_head_axis[T](tree: T, axis: int) -> T:
    """Shift every leaf's ``axis`` left by one, duplicating the last slot into the vacated one."""

    def shift(x: Array) -> Array:
        dst = (slice(None),) * axis + (slice(None, -1),)
        src = (slice(None),) * axis + (slice(1, None),)
        return x.at[dst].set(x[src])

    return jax.tree_util.tree_map(shift, tree)


def shift_head_readout(
    variables: VariableDict, *, kernel_name: str, kernel_axis: int, bias_name: str | None = None
) -> VariableDict:
    """Rotate a shared multi-head readout's head axis, advancing a Bellman-iteration chain one step.

    Shifts the named kernel (and, if given, bias) leaf of a Flax ``{"params": ...}`` variable
    dict along its head axis via :func:`shift_head_axis`; every other parameter (e.g. a shared
    trunk the readout sits on) is left untouched.
    """
    params = variables["params"]
    new_params = {**params, kernel_name: shift_head_axis(params[kernel_name], axis=kernel_axis)}
    if bias_name is not None:
        new_params[bias_name] = shift_head_axis(params[bias_name], axis=0)
    return {**variables, "params": new_params}
