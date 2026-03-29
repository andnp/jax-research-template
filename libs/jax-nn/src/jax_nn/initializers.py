"""Stable orthogonal initializers for RL networks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

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


def legacy_dqn_bound(num_input_units: int) -> float:
    """Return the original DQN uniform bound for a layer.

    The legacy DQN / DQN Zoo initialization samples weights and biases from
    :math:`[-c, c]` where :math:`c = \\sqrt{1 / n}` and :math:`n` is the number of
    input units feeding the layer.

    Args:
        num_input_units: Number of incoming units for the layer.
    """
    if num_input_units <= 0:
        raise ValueError("num_input_units must be positive.")
    return math.sqrt(1.0 / float(num_input_units))


def legacy_dqn_uniform(
    *,
    num_input_units: int | None = None,
    column_axis: int = -1,
    dtype: jnp.dtype = jnp.float32,
) -> jax.nn.initializers.Initializer:
    """Uniform initializer matching the original DQN implementation.

    Samples from :math:`[-c, c]` with :math:`c = \\sqrt{1 / n}` where :math:`n`
    is the number of input units. For kernels, ``n`` is inferred from ``shape``
    by multiplying every axis except ``column_axis``. Bias vectors are 1D, so
    they must pass ``num_input_units`` explicitly.

    Args:
        num_input_units: Optional explicit incoming width. Required for 1D bias
            tensors because their shape alone does not encode fan-in.
        column_axis: Axis treated as the output / column dimension when inferring
            the number of input units from a kernel shape.
        dtype: Desired dtype of the initialized array.
    """

    def init(
        key: Array,
        shape: Sequence[int],
        dtype: Any | None = dtype,
        out_sharding: Any = None,
    ) -> Array:
        del out_sharding
        inferred_num_input_units = _legacy_dqn_num_input_units(
            shape=shape,
            column_axis=column_axis,
            num_input_units=num_input_units,
        )
        bound = legacy_dqn_bound(inferred_num_input_units)
        return jax.random.uniform(
            key,
            shape,
            dtype=dtype,
            minval=-bound,
            maxval=bound,
        )

    return init


def _legacy_dqn_num_input_units(
    *,
    shape: Sequence[int],
    column_axis: int,
    num_input_units: int | None,
) -> int:
    if num_input_units is not None:
        if num_input_units <= 0:
            raise ValueError("num_input_units must be positive.")
        return num_input_units

    if not shape:
        raise ValueError("shape must be non-empty.")

    resolved_column_axis = column_axis if column_axis >= 0 else len(shape) + column_axis
    if resolved_column_axis < 0 or resolved_column_axis >= len(shape):
        raise ValueError("column_axis is out of bounds for shape.")

    if len(shape) == 1:
        raise ValueError("1D bias shapes require an explicit num_input_units value.")

    input_units = 1
    for axis, size in enumerate(shape):
        if axis != resolved_column_axis:
            input_units *= size
    return input_units
