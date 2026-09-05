"""L2 regularization over a named sub-tree of a flax params dict."""

from __future__ import annotations

import jax.numpy as jnp
from flax.typing import VariableDict
from jax_utils.pytree import tree_sum_of_squares


def l2_regularizer(params: VariableDict, head_key: str) -> jnp.ndarray:
    """Sum of squared parameters under ``head_key`` (0 if absent)."""
    params_dict = params.get("params", params)
    head_params = params_dict.get(head_key, {})
    if not head_params:
        return jnp.array(0.0)
    return tree_sum_of_squares(head_params)
