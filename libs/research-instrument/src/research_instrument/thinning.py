"""On-device thinning of one run's per-step metric, usable under ``jit``/``vmap``."""

import jax
import jax.numpy as jnp


def thin_metric(x: jax.Array, *, ema: float | None = None, every: int = 1) -> jax.Array:
    """Smooth then subsample a 1-D per-step metric.

    ``ema`` is an exponential moving average (``z_0 = 0``, no bias
    correction); ``every`` keeps steps 0, every, 2*every, ...
    """
    if ema is not None and not 0 <= ema < 1:
        raise ValueError(f"ema must be in [0, 1), got {ema}")
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if ema is not None:
        x = x.astype(jnp.promote_types(x.dtype, jnp.float32))

        def step(z: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
            z = ema * z + (1 - ema) * v
            return z, z

        _, x = jax.lax.scan(step, jnp.zeros((), x.dtype), x)
    return x[::every]
