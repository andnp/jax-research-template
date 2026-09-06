"""JAX-native replay diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def priority_utilization(priorities: jax.Array, *, epsilon: float = 1e-8) -> jax.Array:
    """Return normalized Shannon entropy for a priority distribution."""
    values = jnp.asarray(priorities, dtype=jnp.float32)
    if values.ndim != 1:
        raise ValueError("priorities must be a one-dimensional vector")
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


def _step_difference(current: jax.Array, sampled: jax.Array) -> jax.Array:
    if jnp.issubdtype(current.dtype, jnp.integer) and jnp.issubdtype(sampled.dtype, jnp.integer):
        common_unsigned = jnp.dtype("uint64" if jax.config.x64_enabled else "uint32")

        def split_sign(value: jax.Array) -> tuple[jax.Array, jax.Array]:
            bits = min(value.dtype.itemsize * 8, 64 if jax.config.x64_enabled else 32)
            local_unsigned = jnp.dtype(f"uint{bits}")
            bits = jnp.asarray(value, dtype=local_unsigned)
            if jnp.issubdtype(value.dtype, jnp.signedinteger):
                negative = value < 0
                magnitude = jnp.where(negative, jnp.bitwise_not(bits) + 1, bits)
            else:
                negative = jnp.zeros_like(value, dtype=bool)
                magnitude = bits
            return negative, jnp.asarray(magnitude, dtype=common_unsigned)

        current_negative, current_magnitude = split_sign(current)
        sampled_negative, sampled_magnitude = split_sign(sampled)
        same_sign = current_negative == sampled_negative
        nonnegative = jnp.where(
            same_sign,
            jnp.where(
                current_negative,
                current_magnitude <= sampled_magnitude,
                current_magnitude >= sampled_magnitude,
            ),
            ~current_negative,
        )
        same_sign_magnitude = jnp.where(
            nonnegative,
            jnp.where(current_negative, sampled_magnitude - current_magnitude, current_magnitude - sampled_magnitude),
            jnp.where(current_negative, current_magnitude - sampled_magnitude, sampled_magnitude - current_magnitude),
        )
        magnitude_float = jnp.where(
            same_sign,
            jnp.asarray(same_sign_magnitude, dtype=jnp.float32),
            jnp.asarray(current_magnitude, dtype=jnp.float32)
            + jnp.asarray(sampled_magnitude, dtype=jnp.float32),
        )
        return jnp.where(nonnegative, magnitude_float, -magnitude_float)

    return jnp.asarray(current, dtype=jnp.float32) - jnp.asarray(sampled, dtype=jnp.float32)


def sample_staleness(
    current_step: jax.Array,
    sampled_steps: jax.Array,
    valid_mask: jax.Array | None = None,
) -> jax.Array:
    """Return the mean age of valid sampled transitions."""
    current = jnp.asarray(current_step)
    sampled = jnp.asarray(sampled_steps)
    mask = None
    if valid_mask is not None:
        mask = jnp.asarray(valid_mask, dtype=bool)
        if mask.shape != sampled.shape:
            raise ValueError("valid_mask must match sampled_steps shape")
    if sampled.size == 0:
        return jnp.float32(0.0)

    ages = _step_difference(current, sampled)
    if mask is None:
        return jnp.mean(ages, dtype=jnp.float32)

    count = jnp.sum(mask, dtype=jnp.float32)
    total = jnp.sum(jnp.where(mask, ages, 0.0), dtype=jnp.float32)
    safe_count = jnp.where(count > 0.0, count, 1.0)
    return jnp.where(count > 0.0, total / safe_count, 0.0)
