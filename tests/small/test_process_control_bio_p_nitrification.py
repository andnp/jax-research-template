from collections.abc import Callable

import jax
import jax.numpy as jnp
from process_control.benchmarks.bio_p import BioPConfig, make_bio_p_benchmark
from process_control.benchmarks.combined_np import CombinedNPConfig, make_combined_np_benchmark

_INFLUENT_NH4 = 31.56  # BioPConfig's make_default_influent_asm2d, index 9 (S_NH, g N/m³)


def _rollout(reset: Callable, step: Callable, action: jax.Array, n_steps: int, seed: int = 0):
    state, _ = reset(jax.random.PRNGKey(seed))

    def scan(state: object, key: jax.Array):
        state, obs, reward, done, info = step(state, action, key)
        return state, info

    def run_scan(state: object):
        return jax.lax.scan(scan, state, jax.random.split(jax.random.PRNGKey(seed + 1), n_steps))

    return jax.jit(run_scan)(state)


class TestBioPNitrification:
    def test_default_aeration_nitrifies_below_influent(self) -> None:
        """At default aeration, SRT ~12 d comfortably exceeds the ASM1
        nitrifier washout SRT (~2.2 d from mu_A=0.5/d, b_A=0.05/d), so a
        10-day rollout should drive effluent NH4 well below the influent
        value rather than drifting toward it."""
        config = BioPConfig()
        reset, step = make_bio_p_benchmark(config)
        action = jnp.array([0.5, 0.5])
        n_steps = 10 * config.steps_per_day

        _, info = _rollout(reset, step, action, n_steps)
        nh4_eff = info["nh4_eff"]

        assert bool(jnp.all(jnp.isfinite(nh4_eff)))
        assert float(nh4_eff[-1]) < 0.5 * _INFLUENT_NH4

    def test_no_aeration_degrades_nitrification(self) -> None:
        """With aeration off, autotrophs get no oxygen (K_OA-limited to
        ~zero) regardless of SRT, so effluent NH4 should track the influent
        value instead of being removed."""
        config = BioPConfig(kla_aerobic=0.0)
        reset, step = make_bio_p_benchmark(config)
        action = jnp.array([0.5, 0.5])
        n_steps = 10 * config.steps_per_day

        _, info = _rollout(reset, step, action, n_steps)
        nh4_eff = info["nh4_eff"]

        assert bool(jnp.all(jnp.isfinite(nh4_eff)))
        assert float(nh4_eff[-1]) > 0.8 * _INFLUENT_NH4


class TestCombinedNPNitrification:
    def test_full_aeration_nitrifies_below_influent(self) -> None:
        """With the aerobic zone fully aerated, nitrifiers outgrow washout at the ~12 d SRT,
        so a 10-day rollout should drive effluent NH4 well below the influent value."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        action = jnp.array([1.0, 0.5, 0.0, 0.0])
        n_steps = 10 * config.steps_per_day

        _, info = _rollout(reset, step, action, n_steps)
        nh4_eff = info["nh4_eff"]

        assert bool(jnp.all(jnp.isfinite(nh4_eff)))
        assert float(nh4_eff[-1]) < 0.5 * _INFLUENT_NH4

    def test_no_aeration_degrades_nitrification(self) -> None:
        """With aeration off, autotrophs get no oxygen regardless of SRT, so effluent NH4
        should track the influent value instead of being removed."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        action = jnp.array([0.0, 0.5, 0.0, 0.0])
        n_steps = 10 * config.steps_per_day

        _, info = _rollout(reset, step, action, n_steps)
        nh4_eff = info["nh4_eff"]

        assert bool(jnp.all(jnp.isfinite(nh4_eff)))
        assert float(nh4_eff[-1]) > 0.8 * _INFLUENT_NH4
