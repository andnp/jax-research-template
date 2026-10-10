"""Behavioural tests for the membrane_fouling benchmark."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.membrane_fouling import (
    MembraneFoulingBenchState,
    MembraneFoulingConfig,
    make_membrane_fouling_benchmark,
)
from process_control.units.membrane import MembraneParams


def _first_step_info(action: list[float]):
    config = MembraneFoulingConfig()
    reset, step = make_membrane_fouling_benchmark(config)
    state, _ = reset(jax.random.PRNGKey(0))
    return config, step(state, jnp.array(action), jax.random.PRNGKey(1))[4]


def _first_step_reward(action: list[float]):
    reset, step = make_membrane_fouling_benchmark(MembraneFoulingConfig())
    state, _ = reset(jax.random.PRNGKey(0))
    return float(step(state, jnp.array(action), jax.random.PRNGKey(1))[2])


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


class TestOperatingCosts:
    def test_scour_energy_follows_intensity(self) -> None:
        """Full scour draws scour_kw_per_m2 * area * dt of energy; no scour draws none."""
        _, idle = _first_step_info([0.5, 0.0, 0.0])
        _, scoured = _first_step_info([0.5, 1.0, 0.0])

        assert float(idle["scour_energy_kwh"]) == pytest.approx(0.0)
        assert float(scoured["scour_energy_kwh"]) == pytest.approx(0.02 * 100.0 * 0.01)

    def test_pumping_energy_is_hydraulic_work_over_efficiency(self) -> None:
        """Pump energy equals TMP (Pa) * permeate flow (m3/h) * dt (h) / efficiency, in kWh."""
        config, info = _first_step_info([1.0, 0.0, 0.0])

        joules = float(info["tmp"]) * float(info["q_permeate"]) * config.dt
        assert float(info["pump_energy_kwh"]) == pytest.approx(joules / config.pump_eff / 3.6e6)
        assert float(info["pump_energy_kwh"]) > 0.0

    def test_backwash_water_volume_uses_wash_flux_and_duration(self) -> None:
        """A wash of 0.03 h at twice the 0.05 m/h flux over 100 m2 uses 0.3 m3, charged once."""
        config = MembraneFoulingConfig(membrane=MembraneParams(bw_duration=0.03))
        reset, step = make_membrane_fouling_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        wash = jnp.array([0.5, 0.0, 1.0])

        state, _, _, _, first = step(state, wash, jax.random.PRNGKey(1))
        _, _, _, _, held = step(state, wash, jax.random.PRNGKey(2))

        assert float(first["backwash_water_m3"]) == pytest.approx(0.3)
        assert float(held["backwash_water_m3"]) == pytest.approx(0.0)


class TestDemandReward:
    def test_surplus_production_earns_nothing_and_shortfall_is_penalised(self) -> None:
        """Producing above demand costs only energy; producing below it loses the unmet water value."""
        _, info = _first_step_info([0.0, 0.0, 0.0])
        # Step 0 demand is the diurnal trough: 0.45 of the 5 m3/h capacity.
        assert float(info["demand"]) == pytest.approx(2.25)

        surplus = _first_step_reward([1.0, 0.0, 0.0])
        shortfall = _first_step_reward([0.0, 0.0, 0.0])

        # 1.0 m3/h produced against 2.25 m3/h demanded: (1.25 * 0.5 * 0.01) / (0.5 * 5 * 0.01)
        assert shortfall == pytest.approx(-0.25, abs=0.01)
        assert -0.01 < surplus <= 0.0

    def test_demand_observation_follows_the_diurnal_cycle(self) -> None:
        """The demand observation is the next step's fraction of capacity: trough at day start, peak mid-day."""
        config = MembraneFoulingConfig()
        reset, step = make_membrane_fouling_benchmark(config)
        state, obs = reset(jax.random.PRNGKey(0))
        assert float(obs[5]) == pytest.approx(config.demand_mean - config.demand_amp)

        before_noon = replace(state, step_count=jnp.array(config.steps_per_day // 2 - 1, dtype=jnp.int32))
        _, noon_obs, _, _, _ = step(before_noon, jnp.array([0.5, 0.0, 0.0]), jax.random.PRNGKey(1))
        assert float(noon_obs[5]) == pytest.approx(config.demand_mean + config.demand_amp)
