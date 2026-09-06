"""JAX-native diagnostics for learned representations."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def dead_neurons(activations: jax.Array, *, atol: float = 0.0) -> jax.Array:
    """Return the fraction of units inactive across all observation axes."""
    values = jnp.asarray(activations, dtype=jnp.float32)
    values = jnp.atleast_2d(values)
    if values.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    dead = jnp.all(jnp.abs(values) <= atol, axis=tuple(range(values.ndim - 1)))
    return jnp.mean(dead.astype(jnp.float32))


def feature_drift(old_features: jax.Array, new_features: jax.Array, *, epsilon: float = 1e-8) -> jax.Array:
    """Return mean row-wise cosine distance across leading observation axes."""
    old = jnp.asarray(old_features, dtype=jnp.float32)
    new = jnp.asarray(new_features, dtype=jnp.float32)
    if old.shape != new.shape:
        raise ValueError("old_features and new_features must have the same shape")
    if old.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    old_norm = jnp.linalg.norm(old, axis=-1)
    new_norm = jnp.linalg.norm(new, axis=-1)
    denominator = old_norm * new_norm
    valid = (old_norm > epsilon) & (new_norm > epsilon)
    safe_denominator = jnp.where(valid, denominator, 1.0)
    similarity = jnp.where(
        valid,
        jnp.sum(old * new, axis=-1) / safe_denominator,
        0.0,
    )
    distance = 1.0 - jnp.clip(similarity, -1.0, 1.0)
    return jnp.mean(distance)
