"""Exploration-rate schedules shared across epsilon-greedy agents."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def linear_epsilon_schedule(
    step_index: Array,
    total_timesteps: Array,
    epsilon_start: Array,
    epsilon_end: Array,
    epsilon_fraction: Array,
) -> Array:
    """Linearly decay epsilon from ``epsilon_start`` to ``epsilon_end`` over the first ``epsilon_fraction`` of ``total_timesteps``, then hold at ``epsilon_end``."""
    return jnp.maximum(
        epsilon_end,
        epsilon_start - (epsilon_start - epsilon_end) * (step_index / (total_timesteps * epsilon_fraction)),
    ).astype(jnp.float32)
