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


def linear_probe_r2(
    features: jax.Array,
    targets: jax.Array,
    *,
    ridge: float = 1e-4,
    fit_intercept: bool = True,
    epsilon: float = 1e-8,
) -> jax.Array:
    """Return mean R² from a ridge-regularized linear probe."""
    x = jnp.asarray(features, dtype=jnp.float32)
    y = jnp.asarray(targets, dtype=jnp.float32)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("features and targets must have observation and output axes")
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError("features and targets must have matching observation shapes")
    if x.size == 0 or y.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    x = x.reshape((-1, x.shape[-1]))
    y = y.reshape((-1, y.shape[-1]))
    if fit_intercept:
        x_fit = x - jnp.mean(x, axis=0, keepdims=True)
        y_fit = y - jnp.mean(y, axis=0, keepdims=True)
    else:
        x_fit = x
        y_fit = y

    gram = x_fit.T @ x_fit + ridge * jnp.eye(x_fit.shape[-1], dtype=x_fit.dtype)
    weights = jax.scipy.linalg.solve(gram, x_fit.T @ y_fit)
    residual = y_fit - x_fit @ weights
    ss_res = jnp.sum(residual**2, axis=0)
    centered_targets = y - jnp.mean(y, axis=0, keepdims=True)
    ss_tot = jnp.sum(centered_targets**2, axis=0)
    valid = ss_tot > epsilon
    safe_ss_tot = jnp.where(valid, ss_tot, 1.0)
    scores = jnp.where(valid, 1.0 - ss_res / safe_ss_tot, 0.0)
    return jnp.mean(scores)
