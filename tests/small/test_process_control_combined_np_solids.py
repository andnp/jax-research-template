import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from process_control.benchmarks.combined_np import CombinedNPConfig, make_combined_np_benchmark

# ASM2d S_ALK component index in the aerobic zone state (combined_np.py precipitation step)
_S_ALK_INDEX = 12
_S_PO4_INDEX = 13


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
        """A moderate FeCl3 dose leaves less aerobic alkalinity than no dosing."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = config.steps_per_day

        state_no_fe, _ = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 0.3]), n_steps, seed=20)
        state_fe, _ = _rollout(reset, step, jnp.array([0.5, 0.5, 0.5, 0.3]), n_steps, seed=20)

        assert float(state_fe.aerobic_state[_S_ALK_INDEX]) < float(state_no_fe.aerobic_state[_S_ALK_INDEX])

    def test_fecl3_beyond_available_phosphate_still_consumes_alkalinity(self) -> None:
        """Iron dosed into nearly phosphate-free liquor is mostly unconsumed by precipitation but still draws alkalinity."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        state, _ = reset(jax.random.PRNGKey(0))
        depleted = dataclasses.replace(state, aerobic_state=state.aerobic_state.at[_S_PO4_INDEX].set(0.0))
        key = jax.random.PRNGKey(1)

        no_fe, _, _, _, info_no_fe = step(depleted, jnp.array([0.5, 0.5, 0.0, 0.3]), key)
        fe, _, _, _, info_fe = step(depleted, jnp.array([0.5, 0.5, 1.0, 0.3]), key)

        applied_fe = config.fe_dose_max * float(info_fe["q_in"]) * config.dt / config.v_aerobic
        assert float(info_fe["fe_consumed"]) < 0.5 * applied_fe
        assert float(fe.aerobic_state[_S_ALK_INDEX]) < float(no_fe.aerobic_state[_S_ALK_INDEX])


class TestCombinedNPChemicalPhosphorusSink:
    def test_dosed_iron_lowers_phosphate_in_reactor_state(self) -> None:
        """Precipitated P leaves the aerobic reactor state, not just the reported effluent value."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = config.steps_per_day

        state_no_fe, info_no_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 0.3]), n_steps, seed=20)
        state_fe, info_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 0.5, 0.3]), n_steps, seed=20)

        assert float(state_fe.aerobic_state[_S_PO4_INDEX]) == float(info_fe["po4_eff"][-1])
        assert float(state_fe.aerobic_state[_S_PO4_INDEX]) < float(state_no_fe.aerobic_state[_S_PO4_INDEX])
        assert float(info_fe["p_captured_kg"].sum()) > float(info_no_fe["p_captured_kg"].sum()) == 0.0

    def test_dosed_iron_lowers_anaerobic_phosphate_through_recycle(self) -> None:
        """Captured P stays captured, so the RAS carries less phosphate to the anaerobic zone."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = config.steps_per_day

        _, info_no_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 0.3]), n_steps, seed=20)
        _, info_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 0.5, 0.3]), n_steps, seed=20)

        assert float(info_fe["po4_anaerobic"][-1]) < float(info_no_fe["po4_anaerobic"][-1])

    def test_maximum_iron_lowers_final_effluent_phosphate(self) -> None:
        """With everything else fixed, the maximum FeCl3 dose ends a day with lower effluent PO4 than none."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = config.steps_per_day

        _, info_no_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 0.0, 0.3]), n_steps, seed=20)
        _, info_max_fe = _rollout(reset, step, jnp.array([0.5, 0.5, 1.0, 0.3]), n_steps, seed=20)

        assert float(info_max_fe["po4_eff"][-1]) < float(info_no_fe["po4_eff"][-1])
