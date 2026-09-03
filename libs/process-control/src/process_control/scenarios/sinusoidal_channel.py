"""Reusable sinusoidal channel: diurnal waveform + random-walk drift + noise.

Provides the common signal generation pattern shared by DiurnalSource,
GasSource, and any future scenario source.  The caller supplies the
diurnal waveform shape (sin, sin+cos, etc.) so no particular shape is
prescribed.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ChannelParams:
    mean: float
    amplitude: float
    min_value: float
    max_value: float
    noise_std: float = 0.0
    drift_scale: float = 0.0
    drift_clip: float = 50.0


def channel_step(
    drift: jax.Array,
    diurnal_signal: jax.Array,
    params: ChannelParams,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute one channel value: mean + amplitude × diurnal + drift + noise.

    The caller provides ``diurnal_signal`` (e.g. ``sin(phase)``).
    This function handles drift, noise, and clipping.

    Returns:
        (new_drift, value)
    """
    k1, k2 = jax.random.split(rng_key)
    new_drift = jnp.clip(
        drift + jax.random.normal(k1) * params.drift_scale,
        -params.drift_clip,
        params.drift_clip,
    )
    noise = jax.random.normal(k2) * params.noise_std
    value = jnp.clip(
        params.mean + params.amplitude * diurnal_signal + new_drift + noise,
        params.min_value,
        params.max_value,
    )
    return new_drift, value
