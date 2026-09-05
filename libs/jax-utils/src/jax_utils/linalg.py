from __future__ import annotations

import jax
import jax.numpy as jnp


def stable_rank(features: jax.Array) -> jax.Array:
    """Stable rank of a feature covariance matrix: tr(C) / lambda_max(C)."""
    cov = jnp.cov(features, rowvar=False)
    cov = jnp.atleast_2d(cov)
    evals = jnp.linalg.eigvalsh(cov)
    evals = jnp.maximum(evals, 1e-8)
    return jnp.sum(evals) / jnp.max(evals)
