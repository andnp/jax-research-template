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

    def test_backwash_water_price_is_charged_once_per_wash_start(self) -> None:
        """The reward drops by bw_water_price * wash volume (normalized) on a wash start, not while held."""
        config = MembraneFoulingConfig(membrane=MembraneParams(bw_duration=0.03), bw_water_price=0.5)
        free = replace(config, bw_water_price=0.0)
        wash = jnp.array([0.5, 0.0, 1.0])

        def rewards(cfg: MembraneFoulingConfig) -> tuple[float, float]:
            reset, step = make_membrane_fouling_benchmark(cfg)
            state, _ = reset(jax.random.PRNGKey(0))
            state, _, first, _, _ = step(state, wash, jax.random.PRNGKey(1))
            _, _, held, _, _ = step(state, wash, jax.random.PRNGKey(2))
            return float(first), float(held)

        first_priced, held_priced = rewards(config)
        first_free, held_free = rewards(free)

        ref_value = config.water_value * config.flux_max * config.membrane.area * config.dt
        assert first_free - first_priced == pytest.approx(0.5 * 0.3 / ref_value, rel=1e-4)
        assert held_free == pytest.approx(held_priced)


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


def _peak_tmp(wash_every: int | None, steps: int) -> float:
    config = MembraneFoulingConfig()
    reset, step = make_membrane_fouling_benchmark(config)
    state, _ = reset(jax.random.PRNGKey(0))

    def body(carry: MembraneFoulingBenchState, inputs: tuple[jax.Array, jax.Array]):
        key, index = inputs
        wash = 0.0 if wash_every is None else (index % wash_every == 0) * 1.0
        new_state, _, _, _, info = step(carry, jnp.array([0.5, 0.0, wash]), key)
        return new_state, info["tmp"]

    keys = jax.random.split(jax.random.PRNGKey(1), steps)
    _, tmp = jax.lax.scan(body, state, (keys, jnp.arange(steps)))
    return float(tmp.max())


class TestCalibratedFouling:
    def test_unwashed_membrane_passes_tmp_target_within_a_day_but_timer_keeps_it_below(self) -> None:
        """Without backwash TMP crosses the soft target in a day; a periodic backwash holds it under."""
        config = MembraneFoulingConfig()

        assert _peak_tmp(None, config.steps_per_day) > config.tmp_target
        assert _peak_tmp(300, config.steps_per_day) < config.tmp_target


def _fouled_state(r_reversible: float, r_irreversible: float = 0.0, wash_remaining: float = 0.0):
    config = MembraneFoulingConfig()
    reset, step = make_membrane_fouling_benchmark(config)
    state, _ = reset(jax.random.PRNGKey(0))
    membrane = replace(
        state.membrane_state,
        r_reversible=jnp.array(r_reversible),
        r_irreversible=jnp.array(r_irreversible),
        is_backwashing=jnp.array(wash_remaining > 0.0),
        backwash_remaining=jnp.array(wash_remaining),
    )
    return config, step, replace(state, membrane_state=membrane)


class TestForcedCip:
    def test_trip_during_partial_wash_step_uses_operating_pressure(self) -> None:
        """A step that is half wash, half permeating still trips at the limit, charges the CIP, and bills pumping on operating TMP."""
        config, step, state = _fouled_state(5e13, 1e13, wash_remaining=0.005)

        new_state, _, reward, _, info = step(state, jnp.array([1.0, 0.0, 0.0]), jax.random.PRNGKey(1))

        assert float(info["tmp"]) >= config.membrane.max_tmp
        assert bool(info["tripped"])
        assert float(new_state.cip_remaining) == pytest.approx(config.cip_duration)
        assert float(new_state.membrane_state.r_reversible) == pytest.approx(0.0)
        assert float(new_state.membrane_state.r_irreversible) < 0.25e13
        half_step_flow = config.flux_max * config.membrane.area / 2.0
        joules = float(info["tmp"]) * half_step_flow * config.dt
        assert float(info["pump_energy_kwh"]) == pytest.approx(joules / config.pump_eff / 3.6e6, rel=1e-5)
        assert float(reward) < -config.cip_chem_cost / (config.water_value * 5.0 * config.dt)

    def test_membrane_is_offline_for_the_cip_duration_then_resumes(self) -> None:
        """No flow, scour, or wash happens while the CIP runs; production resumes when it ends."""
        config, step, state = _fouled_state(0.0)
        state = replace(state, cip_remaining=jnp.array(config.dt))
        command = jnp.array([1.0, 1.0, 1.0])

        state, obs, _, _, offline = step(state, command, jax.random.PRNGKey(1))
        _, _, _, _, resumed = step(state, jnp.array([1.0, 1.0, 0.0]), jax.random.PRNGKey(2))

        assert float(offline["q_permeate"]) == pytest.approx(0.0)
        assert float(offline["scour_energy_kwh"]) == pytest.approx(0.0)
        assert float(offline["backwash_water_m3"]) == pytest.approx(0.0)
        assert float(resumed["q_permeate"]) > 0.0
        assert float(obs[6]) == pytest.approx(0.0)

    def test_unwashed_membrane_trips_and_timer_prevents_it(self) -> None:
        """Over five days at mid flux an unwashed membrane trips repeatedly while a 3 h backwash timer never does."""
        config = MembraneFoulingConfig()
        reset, step = make_membrane_fouling_benchmark(config)
        steps = 5 * config.steps_per_day

        def trips(wash_every: int | None) -> int:
            def body(carry: MembraneFoulingBenchState, inputs: tuple[jax.Array, jax.Array]):
                key, index = inputs
                wash = 0.0 if wash_every is None else (index % wash_every == 0) * 1.0
                new_state, _, _, _, info = step(carry, jnp.array([0.5, 0.0, wash]), key)
                return new_state, info["tripped"]

            state, _ = reset(jax.random.PRNGKey(0))
            keys = jax.random.split(jax.random.PRNGKey(1), steps)
            _, tripped = jax.lax.scan(body, state, (keys, jnp.arange(steps)))
            return int(tripped.sum())

        assert trips(None) > 0
        assert trips(300) == 0


class TestFeedVariation:
    def test_feed_drift_wanders_beyond_the_diurnal_amplitude_within_its_clip(self) -> None:
        """Over days the feed TSS drift exceeds the diurnal amplitude, so feed quality is not fixed, yet stays within its clip."""
        config = MembraneFoulingConfig()
        reset, step = make_membrane_fouling_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        action = jnp.array([0.5, 0.0, 0.0])

        def body(carry: MembraneFoulingBenchState, key: jax.Array):
            new_state, _, _, _, _ = step(carry, action, key)
            return new_state, new_state.tss_drift

        _, drift = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(1), 5 * config.steps_per_day))

        assert float(jnp.abs(drift).max()) > config.feed_tss_amplitude
        assert float(jnp.abs(drift).max()) <= config.feed_tss_drift_clip
