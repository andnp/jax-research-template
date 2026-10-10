"""Behavioural tests for the membrane_fouling benchmark."""

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.membrane_fouling import (
    MembraneFoulingBenchState,
    MembraneFoulingConfig,
    make_membrane_fouling_benchmark,
)


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


class TestObservationScaling:
    def test_observations_stay_normalised_without_backwash(self) -> None:
        """Backwash age and feed TSS are bounded to [0, 1] even when no wash ever happens."""
        config = MembraneFoulingConfig()
        reset, step = make_membrane_fouling_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([0.5, 0.0, 0.0])
        steps = int(1.5 * config.bw_age_ref / config.dt)

        def body(carry: MembraneFoulingBenchState, key: jax.Array):
            new_state, obs, _, _, _ = step(carry, action, key)
            return new_state, obs

        _, observations = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(1), steps))

        assert float(observations[:, 3].max()) == pytest.approx(1.0)
        assert float(observations[:, 3:].min()) >= 0.0
        assert float(observations[:, 3:].max()) <= 1.0
