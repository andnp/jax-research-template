"""JAX-native replay diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def priority_utilization(priorities: jax.Array, *, epsilon: float = 1e-8) -> jax.Array:
    """Return normalized Shannon entropy for a priority distribution."""
    values = jnp.asarray(priorities, dtype=jnp.float32)
    capacity = values.shape[0]
    if capacity <= 1:
        return jnp.float32(0.0)

    total = jnp.sum(values, dtype=jnp.float32)
    safe_total = jnp.where(total > 0.0, total, 1.0)
    probabilities = values / safe_total
    safe_probabilities = jnp.maximum(probabilities, jnp.float32(epsilon))
    terms = jnp.where(
        probabilities > 0.0,
        -probabilities * jnp.log(safe_probabilities),
        0.0,
    )
    return jnp.sum(terms, dtype=jnp.float32) / jnp.log(jnp.float32(capacity))


def sample_staleness(
    current_step: jax.Array,
    sampled_steps: jax.Array,
    valid_mask: jax.Array | None = None,
) -> jax.Array:
    """Return the mean age of valid sampled transitions."""
    current = jnp.asarray(current_step, dtype=jnp.float32)
    sampled = jnp.asarray(sampled_steps, dtype=jnp.float32)
    mask = None
    if valid_mask is not None:
        mask = jnp.asarray(valid_mask, dtype=bool)
        if mask.shape != sampled.shape:
            raise ValueError("valid_mask must match sampled_steps shape")
    if sampled.size == 0:
        return jnp.float32(0.0)

    ages = current - sampled
    if mask is None:
        return jnp.mean(ages, dtype=jnp.float32)

    count = jnp.sum(mask, dtype=jnp.float32)
    total = jnp.sum(jnp.where(mask, ages, 0.0), dtype=jnp.float32)
    safe_count = jnp.where(count > 0.0, count, 1.0)
    return jnp.where(count > 0.0, total / safe_count, 0.0)
