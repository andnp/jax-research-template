"""Small (unit) tests for linear_epsilon_schedule."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from rl_components.exploration import linear_epsilon_schedule


class TestLinearEpsilonScheduleBoundaries:
    def test_start_value_at_step_zero(self) -> None:
        epsilon = linear_epsilon_schedule(
            step_index=jnp.asarray(0.0),
            total_timesteps=jnp.asarray(1_000.0),
            epsilon_start=jnp.asarray(1.0),
            epsilon_end=jnp.asarray(0.05),
            epsilon_fraction=jnp.asarray(0.5),
        )
        assert float(epsilon) == 1.0

    def test_holds_at_end_value_past_decay_window(self) -> None:
        epsilon = linear_epsilon_schedule(
            step_index=jnp.asarray(1_000.0),
            total_timesteps=jnp.asarray(1_000.0),
            epsilon_start=jnp.asarray(1.0),
            epsilon_end=jnp.asarray(0.05),
            epsilon_fraction=jnp.asarray(0.5),
        )
        assert float(epsilon) == pytest.approx(0.05)

    def test_midpoint_of_decay_window(self) -> None:
        epsilon = linear_epsilon_schedule(
            step_index=jnp.asarray(250.0),
            total_timesteps=jnp.asarray(1_000.0),
            epsilon_start=jnp.asarray(1.0),
            epsilon_end=jnp.asarray(0.0),
            epsilon_fraction=jnp.asarray(0.5),
        )
        assert float(epsilon) == 0.5


class TestLinearEpsilonScheduleReturnType:
    def test_returns_float32_array(self) -> None:
        epsilon = linear_epsilon_schedule(
            step_index=jnp.asarray(10.0),
            total_timesteps=jnp.asarray(1_000.0),
            epsilon_start=jnp.asarray(1.0),
            epsilon_end=jnp.asarray(0.05),
            epsilon_fraction=jnp.asarray(0.5),
        )
        assert epsilon.dtype == jnp.float32
