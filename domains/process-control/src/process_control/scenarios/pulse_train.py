"""Random rectangular pulses (batch dumps, slugs): geometric start times, exponential durations, uniform amplitudes."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PulseTrainParams:
    """A new pulse starts with probability `rate_per_step` whenever none is active; amplitudes are uniform on [low, high)."""

    rate_per_step: float
    mean_duration_steps: float
    low: float
    high: float


@jax_dataclass
class PulseTrainState:
    remaining: jax.Array
    amplitude: jax.Array


def reset() -> PulseTrainState:
    return PulseTrainState(remaining=jnp.array(0, dtype=jnp.int32), amplitude=jnp.array(0.0))


def step(state: PulseTrainState, params: PulseTrainParams, rng_key: jax.Array) -> tuple[PulseTrainState, jax.Array]:
    """Advance one step, returning the pulse level (zero between pulses). A pulse lasts at least one step."""
    k_start, k_len, k_amp = jax.random.split(rng_key, 3)
    start = (state.remaining <= 0) & (jax.random.uniform(k_start) < params.rate_per_step)
    duration = 1 + jnp.floor(jax.random.exponential(k_len) * params.mean_duration_steps).astype(jnp.int32)
    amplitude = jax.random.uniform(k_amp, minval=params.low, maxval=params.high)
    remaining = jnp.where(start, duration, state.remaining)
    amplitude = jnp.where(start, amplitude, state.amplitude)
    level = jnp.where(remaining > 0, amplitude, 0.0)
    return PulseTrainState(remaining=jnp.maximum(remaining - 1, 0), amplitude=amplitude), level
