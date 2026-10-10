"""Behavioural tests for the membrane_fouling benchmark."""

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.membrane_fouling import MembraneFoulingConfig, make_membrane_fouling_benchmark


def _first_step_info(action: list[float]):
    config = MembraneFoulingConfig()
    reset, step = make_membrane_fouling_benchmark(config)
    state, _ = reset(jax.random.PRNGKey(0))
    return config, step(state, jnp.array(action), jax.random.PRNGKey(1))[4]


class TestFluxAction:
    def test_action_endpoints_command_flux_range_endpoints(self) -> None:
        """Action 0 and 1 give the minimum and maximum operating flux, not zero flux."""
        config, low = _first_step_info([0.0, 0.0, 0.0])
        _, high = _first_step_info([1.0, 0.0, 0.0])

        area = config.membrane.area
        assert float(low["q_permeate"]) == pytest.approx(config.flux_min * area)
        assert float(high["q_permeate"]) == pytest.approx(config.flux_max * area)
