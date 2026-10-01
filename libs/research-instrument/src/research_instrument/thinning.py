"""On-device thinning of one run's per-step metric, usable under ``jit``/``vmap``."""

import jax


def thin_metric(x: jax.Array, *, every: int = 1) -> jax.Array:
    """Keep steps 0, every, 2*every, ... of a 1-D per-step metric."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    return x[::every]
