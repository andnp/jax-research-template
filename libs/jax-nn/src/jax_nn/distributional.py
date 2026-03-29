"""Distributional RL helpers for categorical value distributions."""

from __future__ import annotations

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from jax_nn.initializers import output_orthogonal


def categorical_l2_project(
    target_support: Array,
    target_probabilities: Array,
    support: Array,
) -> Array:
    """Project a categorical distribution onto a fixed atom support.

    Implements the C51 projection operator onto a uniformly spaced support.

    Args:
        target_support: Source atom values, shape ``(..., num_atoms)``.
        target_probabilities: Source probabilities, shape ``(..., num_atoms)``.
        support: Destination support, shape ``(num_atoms,)``.

    Returns:
        Projected probabilities on ``support``, shape ``(..., num_atoms)``.
    """
    support = _validate_support(support)
    chex.assert_equal_shape((target_support, target_probabilities))
    chex.assert_axis_dimension(target_support, -1, support.shape[0])

    dtype = jnp.result_type(target_support, target_probabilities, support)
    support = jnp.asarray(support, dtype=dtype)
    target_support = jnp.asarray(target_support, dtype=dtype)
    target_probabilities = jnp.asarray(target_probabilities, dtype=dtype)

    v_min = support[0]
    v_max = support[-1]
    delta = (v_max - v_min) / jnp.asarray(support.shape[0] - 1, dtype=dtype)

    clipped_support = jnp.clip(target_support, v_min, v_max)
    projection_position = (clipped_support - v_min) / delta
    lower = jnp.floor(projection_position).astype(jnp.int32)
    upper = jnp.ceil(projection_position).astype(jnp.int32)
    same_index = lower == upper

    lower_weight = jnp.where(same_index, 1.0, upper.astype(dtype) - projection_position)
    upper_weight = jnp.where(same_index, 0.0, projection_position - lower.astype(dtype))

    lower_one_hot = jax.nn.one_hot(lower, support.shape[0], dtype=dtype)
    upper_one_hot = jax.nn.one_hot(upper, support.shape[0], dtype=dtype)
    weighted_projection = (
        lower_weight[..., :, None] * lower_one_hot
        + upper_weight[..., :, None] * upper_one_hot
    )
    return jnp.sum(target_probabilities[..., :, None] * weighted_projection, axis=-2)


def categorical_cross_entropy(logits: Array, target_probabilities: Array) -> Array:
    """Return the categorical cross-entropy reduced over the atom axis.

    Args:
        logits: Predicted logits, shape ``(..., num_atoms)``.
        target_probabilities: Target probabilities, shape ``(..., num_atoms)``.
    """
    chex.assert_equal_shape((logits, target_probabilities))

    target_probabilities = jnp.asarray(target_probabilities, dtype=jnp.result_type(logits, target_probabilities))
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target_probabilities * log_probabilities, axis=-1)


def categorical_expected_value(probabilities: Array, support: Array) -> Array:
    """Compute the expected scalar value of a categorical distribution.

    Args:
        probabilities: Atom probabilities, shape ``(..., num_atoms)``.
        support: Atom support, shape ``(num_atoms,)``.
    """
    support = _validate_support(support)
    chex.assert_axis_dimension(probabilities, -1, support.shape[0])
    return jnp.sum(probabilities * support, axis=-1)


class CategoricalValueHead(nn.Module):
    """Linear categorical head that emits ``(action_dim, num_atoms)`` logits."""

    action_dim: int
    num_atoms: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if x.ndim < 1:
            raise ValueError("CategoricalValueHead expects inputs with shape (..., features).")

        logits = nn.Dense(
            self.action_dim * self.num_atoms,
            kernel_init=output_orthogonal(),
            dtype=self.dtype,
        )(jnp.asarray(x, dtype=self.dtype))
        return logits.reshape(logits.shape[:-1] + (self.action_dim, self.num_atoms))


def _validate_support(support: Array) -> Array:
    support = jnp.asarray(support)
    chex.assert_rank(support, 1)
    if support.shape[0] < 2:
        raise ValueError("support must contain at least two atoms.")
    return support