"""Overload behaviour of the anaerobic digester benchmark."""

import jax
import jax.numpy as jnp
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
