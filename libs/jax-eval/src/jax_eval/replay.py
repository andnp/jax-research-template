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
