"""Overload behaviour of the anaerobic digester benchmark."""

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.anaerobic_digester import (
    AnaerobicDigesterBenchState,
    AnaerobicDigesterConfig,
    make_anaerobic_digester_benchmark,
)


def _rollout(config: AnaerobicDigesterConfig, action: jax.Array, days: int):
    reset_fn, step_fn = make_anaerobic_digester_benchmark(config)
    state, _ = reset_fn(jax.random.PRNGKey(0))

    def body(state: AnaerobicDigesterBenchState, key: jax.Array):
        state, _, reward, _, info = step_fn(state, action, key)
        return state, (reward, info["vfa"])

    _, (reward, vfa) = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(1), days * 24))
    return reward, vfa


class TestFeedRange:
    def test_feed_action_endpoints_span_configured_feed_range(self) -> None:
        """Action 0 and 1 map to the configured minimum and maximum feed flow."""
        config = AnaerobicDigesterConfig()
        reset_fn, step_fn = make_anaerobic_digester_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        _, _, _, _, low = step_fn(state, jnp.array([0.0, 0.5]), jax.random.PRNGKey(1))
        _, _, _, _, high = step_fn(state, jnp.array([1.0, 0.5]), jax.random.PRNGKey(1))
        assert float(low["q_feed"]) == config.q_feed_min
        assert float(high["q_feed"]) == config.q_feed_max


class TestOverload:
    def test_full_feed_upsets_digester_while_moderate_feed_does_not(self) -> None:
        """Max feed depletes buffer and drives VFA past the alarm; moderate feed stays below it."""
        config = AnaerobicDigesterConfig()
        _, vfa_full = _rollout(config, jnp.array([1.0, 0.75]), days=60)
        _, vfa_mod = _rollout(config, jnp.array([0.5, 0.75]), days=60)
        assert float(vfa_full.max()) > 3 * config.vfa_limit
        assert float(vfa_mod.max()) < config.vfa_limit

    def test_collapse_penalty_is_bounded_by_vfa_cap(self) -> None:
        """Per-step reward under total collapse stays above the capped-penalty floor."""
        config = AnaerobicDigesterConfig()
        reward, vfa = _rollout(config, jnp.array([1.0, 1.0]), days=90)
        assert float(vfa.max()) > config.vfa_violation_cap * config.vfa_limit
        floor = -config.reward_w_vfa * config.vfa_violation_cap**2 - config.reward_w_ph * 2.0**2
        assert float(reward.min()) > floor - 5.0


class TestRewardUsesTrueState:
    def test_vfa_and_ph_analyzer_noise_does_not_change_reward(self) -> None:
        """Process-health penalties follow the true state, so analyzer noise cannot alter reward."""
        quiet = AnaerobicDigesterConfig(vfa_noise_std=0.0, ph_noise_std=0.0)
        noisy = AnaerobicDigesterConfig(vfa_noise_std=400.0, ph_noise_std=0.5)
        action = jnp.array([1.0, 1.0])
        reward_quiet, vfa = _rollout(quiet, action, days=60)
        reward_noisy, _ = _rollout(noisy, action, days=60)
        assert float(vfa.max()) > quiet.vfa_limit
        assert jnp.allclose(reward_quiet, reward_noisy)

    def test_biogas_analyzer_noise_does_not_change_reward(self) -> None:
        """Production reward follows true biogas flow, so the biogas meter cannot alter reward."""
        quiet = AnaerobicDigesterConfig(biogas_noise_std=0.0)
        noisy = AnaerobicDigesterConfig(biogas_noise_std=500.0)
        action = jnp.array([0.5, 0.75])
        reward_quiet, _ = _rollout(quiet, action, days=10)
        reward_noisy, _ = _rollout(noisy, action, days=10)
        assert jnp.allclose(reward_quiet, reward_noisy)


class TestHeatingCost:
    def test_heating_a_larger_feed_costs_more_at_equal_temperature(self) -> None:
        """With every other term off, the reward gap is the feed-flow share of the sensible-heating proxy."""
        config = AnaerobicDigesterConfig(reward_w_biogas=0.0, reward_w_ch4=0.0, reward_w_vfa=0.0, reward_w_ph=0.0)
        reset_fn, step_fn = make_anaerobic_digester_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        _, _, r_low, _, low = step_fn(state, jnp.array([0.2, 0.5]), jax.random.PRNGKey(1))
        _, _, r_high, _, high = step_fn(state, jnp.array([0.8, 0.5]), jax.random.PRNGKey(1))

        assert float(low["temperature"]) == float(high["temperature"])
        lift = float(low["temperature"]) - config.feed_temperature
        flow_gap = float(high["q_feed"]) - float(low["q_feed"])
        expected_gap = config.reward_w_heat * flow_gap / config.q_feed_max * lift / 20.0
        assert float(r_low) - float(r_high) == pytest.approx(expected_gap, rel=1e-4)

    def test_moderate_feed_out_earns_minimum_feed(self) -> None:
        """Mean reward guards the benchmark scale: moderate feed beats minimum feed."""
        mod, _ = _rollout(AnaerobicDigesterConfig(), jnp.array([0.5, 0.75]), days=60)
        low, _ = _rollout(AnaerobicDigesterConfig(), jnp.array([0.0, 0.75]), days=60)
        assert float(mod.mean()) > float(low.mean())


class TestObservation:
    def test_observation_carries_measured_alkalinity_not_biomass(self) -> None:
        """With a noiseless analyzer, obs[6] tracks alkalinity, which biomass would not."""
        reset_fn, step_fn = make_anaerobic_digester_benchmark(AnaerobicDigesterConfig(alk_noise_std=0.0))
        state, _ = reset_fn(jax.random.PRNGKey(0))
        for _ in range(48):
            state, obs, _, _, info = step_fn(state, jnp.array([0.9, 0.75]), jax.random.PRNGKey(2))
        alk = float(info["alkalinity"])
        assert abs(float(obs[6]) * 5000.0 - alk) < 0.05 * alk

    def test_feed_cod_observation_is_a_lagged_noisy_measurement(self) -> None:
        """obs[5] is an analyzer reading, so it differs from the exact influent COD."""
        reset_fn, step_fn = make_anaerobic_digester_benchmark(AnaerobicDigesterConfig())
        state, _ = reset_fn(jax.random.PRNGKey(0))
        gaps = []
        for i in range(48):
            state, obs, _, _, info = step_fn(state, jnp.array([0.5, 0.75]), jax.random.PRNGKey(i))
            gaps.append(abs(float(obs[5]) * 50000.0 - float(info["feed_cod"])))
        assert max(gaps) > 100.0
