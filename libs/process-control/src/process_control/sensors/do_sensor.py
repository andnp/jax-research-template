"""Dissolved oxygen probe model.

Models a fast-response DO sensor with first-order lag and Gaussian noise.
Typical DO probes have ~30-60 second response times (fast compared to
concentration analyzers which have 5-15 minute cycles).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class DOSensorParams:
    noise_std: float = 0.05  # g O₂/m³ (typical DO probe noise)
    lag_coefficient: float = 0.9  # first-order lag (higher = more smoothing)
    drift_rate: float = 0.001  # drift per step (g O₂/m³)


@jax_dataclass
class DOSensorState:
    held_value: jax.Array
    drift_bias: jax.Array

    @staticmethod
    def create(initial_do: float = 2.0) -> "DOSensorState":
        return DOSensorState(
            held_value=jnp.array(initial_do),
            drift_bias=jnp.array(0.0),
        )


def reset(initial_do: float, rng_key: jax.Array) -> DOSensorState:
    return DOSensorState.create(initial_do)


def step(
    state: DOSensorState,
    true_do: jax.Array,
    params: DOSensorParams,
    rng_key: jax.Array,
) -> tuple[DOSensorState, jax.Array]:
    k1, k2 = jax.random.split(rng_key)

    # Drift random walk
    drift_step = jax.random.normal(k1) * params.drift_rate
    new_drift = state.drift_bias + drift_step

    # Noisy measurement with drift
    noise = jax.random.normal(k2) * params.noise_std
    raw = true_do + noise + new_drift

    # First-order lag
    smoothed = params.lag_coefficient * state.held_value + (1.0 - params.lag_coefficient) * raw

    new_state = DOSensorState(held_value=smoothed, drift_bias=new_drift)
    return new_state, smoothed
