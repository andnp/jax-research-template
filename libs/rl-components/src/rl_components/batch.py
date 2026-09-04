"""Shared execution helpers for dynamic algorithm batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

import jax
import jax.numpy as jnp

ResultT = TypeVar("ResultT")
DynamicValue = bool | float | int


def run_batch(
    train_one: Callable[[Mapping[str, jax.Array], jax.Array], ResultT],
    dynamic_axes: Mapping[str, Sequence[DynamicValue]],
    seeds: Sequence[int],
) -> ResultT:
    """JIT and vmap one algorithm step function over dynamic points and seeds."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds must be integers")

    batch_size = len(seeds)
    if any(len(values) != batch_size for values in dynamic_axes.values()):
        raise ValueError("every dynamic axis must match the number of seeds")

    axes = {name: jnp.asarray(values) for name, values in dynamic_axes.items()}
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    return jax.jit(jax.vmap(train_one))(axes, keys)
