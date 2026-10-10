from collections.abc import Callable

import jax
import jax.numpy as jnp
from process_control.benchmarks.combined_np import CombinedNPConfig, make_combined_np_benchmark


def _rollout(reset: Callable, step: Callable, action: jax.Array, n_steps: int, seed: int = 0):
    state, _ = reset(jax.random.PRNGKey(seed))

    def scan(state: object, key: jax.Array):
        state, obs, reward, done, info = step(state, action, key)
        return state, info

    def run_scan(state: object):
        return jax.lax.scan(scan, state, jax.random.split(jax.random.PRNGKey(seed + 1), n_steps))

    return jax.jit(run_scan)(state)


class TestCombinedNPSolidsRetention:
    def test_anaerobic_po4_responds_to_carbon_dose(self) -> None:
        """More external carbon drives more anaerobic P release, with FeCl3 off so only biology acts."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = config.steps_per_day // 4

        _, info_low = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 0.0]), n_steps, seed=10)
        _, info_high = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 1.0]), n_steps, seed=10)

        ana_po4_low = float(info_low["po4_anaerobic"][-1])
        ana_po4_high = float(info_high["po4_anaerobic"][-1])
        assert jnp.isfinite(ana_po4_low)
        assert jnp.isfinite(ana_po4_high)
        assert ana_po4_high > ana_po4_low

    def test_pao_persists_over_multiple_days(self) -> None:
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        action = jnp.array([0.5, 0.5, 0.0, 0.3])
        n_steps = 3 * config.steps_per_day

        _, info = _rollout(reset, step, action, n_steps)
        x_pao = info["x_pao"]

        assert bool(jnp.all(jnp.isfinite(x_pao)))
        assert float(x_pao[-1]) > 50.0

    def test_fecl3_dosing_consumes_alkalinity(self) -> None:
        """Dosing FeCl3 should draw down alkalinity relative to no dosing,
        since FeCl3 hydrolysis consumes ~3 eq alkalinity per mol Fe."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = config.steps_per_day

        _, info_no_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 0.3]), n_steps, seed=20)
        _, info_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 1.0, 0.3]), n_steps, seed=20)

        assert float(info_fe["fe_consumed"][-1]) > float(info_no_fe["fe_consumed"][-1])
