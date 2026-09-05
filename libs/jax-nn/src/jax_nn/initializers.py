"""Stable orthogonal initializers for RL networks."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_SQRT_2 = math.sqrt(2.0)


def stable_orthogonal(
    scale: float = _SQRT_2,
    column_axis: int = -1,
    dtype: jnp.dtype = jnp.float32,
) -> jax.nn.initializers.Initializer:
    """Orthogonal initializer scaled for ReLU hidden layers.

    Wraps ``jax.nn.initializers.orthogonal`` with a default scale of
    :math:`\\sqrt{2}`, the standard gain for ReLU activations (He et al., 2015).

    Args:
        scale: Multiplicative gain applied after orthogonal init.
        column_axis: Axis treated as the column (output) dimension.
        dtype: Desired dtype of the initialized array.
    """
    return jax.nn.initializers.orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)


def output_orthogonal(
    scale: float = 0.01,
    column_axis: int = -1,
    dtype: jnp.dtype = jnp.float32,
) -> jax.nn.initializers.Initializer:
    """Orthogonal initializer for output / policy layers.

    Uses a small scale (default 0.01) to produce near-uniform initial
    action probabilities and near-zero initial value estimates.

    Args:
        scale: Multiplicative gain (kept small for output layers).
        column_axis: Axis treated as the column (output) dimension.
        dtype: Desired dtype of the initialized array.
    """
    return jax.nn.initializers.orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)
