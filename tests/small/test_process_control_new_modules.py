from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from process_control.benchmarks.anaerobic_digester import (
    AnaerobicDigesterConfig,
    make_anaerobic_digester_benchmark,
)
from process_control.benchmarks.bio_p import BioPConfig, make_bio_p_benchmark
from process_control.benchmarks.bsm1_takacs import BSM1TakacsConfig, make_bsm1_takacs_benchmark
from process_control.benchmarks.combined_np import CombinedNPConfig, make_combined_np_benchmark
from process_control.benchmarks.dewatering import DewateringConfig, make_dewatering_benchmark
from process_control.benchmarks.drinking_water_train import (
    DrinkingWaterTrainConfig,
    make_drinking_water_train_benchmark,
)
from process_control.benchmarks.membrane_fouling import (
    MembraneFoulingConfig,
    make_membrane_fouling_benchmark,
)
from process_control.benchmarks.primary_clarifier import (
    PrimaryClarifierConfig,
    make_primary_clarifier_benchmark,
)
from process_control.benchmarks.reject_water import RejectWaterConfig, make_reject_water_benchmark
from process_control.chemistry.coagulation import CoagulationParams, coagulate
from process_control.units.anaerobic_digester import ADM1Params
from process_control.units.anaerobic_digester import reset as adm1_reset
from process_control.units.anaerobic_digester import step as adm1_step
from process_control.units.asm2d import N_COMPONENTS_ASM2D, ASM2dParams, make_default_influent_asm2d, reactions_asm2d
from process_control.units.dewatering import DewateringParams
from process_control.units.dewatering import reset as dw_reset
from process_control.units.dewatering import step as dw_step
from process_control.units.membrane import MembraneParams, compute_tmp
from process_control.units.membrane import reset as mem_reset
from process_control.units.membrane import step as mem_step
from process_control.units.primary_clarifier import (
    PrimaryClarifierParams,
)
from process_control.units.primary_clarifier import (
    reset as pc_reset,
)
from process_control.units.primary_clarifier import (
    step as pc_step,
)


# ---------------------------------------------------------------------------
# Unit model: Primary Clarifier
# ---------------------------------------------------------------------------
class TestPrimaryClarifierUnit:
    def test_step_reduces_tss(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(0))
        state2, eff_tss, und_tss = pc_step(
            state, jnp.array(250.0), jnp.array(600.0), jnp.array(20.0), params, jnp.array(0.02),
        )
        assert float(eff_tss) < 250.0
        assert float(eff_tss) > 0.0

    def test_underflow_tss_exceeds_effluent(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(1))
        _, eff_tss, und_tss = pc_step(
            state, jnp.array(250.0), jnp.array(600.0), jnp.array(20.0), params, jnp.array(0.02),
        )
        assert float(und_tss) > float(eff_tss)

    def test_sludge_mass_increases_with_no_wasting(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(2))
        state2, _, _ = pc_step(
            state, jnp.array(250.0), jnp.array(600.0), jnp.array(0.0), params, jnp.array(0.02),
        )
        assert float(state2.sludge_mass) >= float(state.sludge_mass)

    def test_jit_compatible(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(3))
        jit_step = jax.jit(lambda s: pc_step(s, jnp.array(250.0), jnp.array(600.0), jnp.array(20.0), params, jnp.array(0.02)))
        state2, eff, und = jit_step(state)
        assert jnp.isfinite(eff)
        assert jnp.isfinite(und)


# ---------------------------------------------------------------------------
# Unit model: Dewatering
# ---------------------------------------------------------------------------
class TestDewateringUnit:
    def test_polymer_increases_capture_and_dryness(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(0))
        feed_tss = jnp.array(10000.0)
        q_feed = jnp.array(30.0)
        dt = jnp.array(0.05)

        _, dry_lo, filt_lo, _ = dw_step(state, feed_tss, q_feed, jnp.array(0.0), jnp.array(0.5), params, dt)
        _, dry_hi, filt_hi, _ = dw_step(state, feed_tss, q_feed, jnp.array(10.0), jnp.array(0.5), params, dt)

        assert float(dry_hi) > float(dry_lo)
        assert float(filt_hi) < float(filt_lo)

    def test_outputs_positive(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(1))
        state2, dryness, filt_tss, q_filt = dw_step(
            state, jnp.array(10000.0), jnp.array(30.0), jnp.array(5.0), jnp.array(0.5), params, jnp.array(0.05),
        )
        assert float(dryness) > 0.0
        assert float(filt_tss) >= 0.0
        assert float(q_filt) >= 0.0

    def test_cumulative_cake_increases(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(2))
        state2, _, _, _ = dw_step(
            state, jnp.array(10000.0), jnp.array(30.0), jnp.array(5.0), jnp.array(0.5), params, jnp.array(0.05),
        )
        assert float(state2.cake_produced) > 0.0

    def test_jit_compatible(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(3))
        jit_step = jax.jit(lambda s: dw_step(s, jnp.array(10000.0), jnp.array(30.0), jnp.array(5.0), jnp.array(0.5), params, jnp.array(0.05)))
        s2, dry, filt, q = jit_step(state)
        assert jnp.isfinite(dry)


# ---------------------------------------------------------------------------
# Unit model: Membrane
# ---------------------------------------------------------------------------
class TestMembraneUnit:
    def test_fouling_increases_tmp(self) -> None:
        params = MembraneParams()
        state = mem_reset(params, jax.random.PRNGKey(0))
        flux = jnp.array(0.03)
        air = jnp.array(0.0)
        dt = jnp.array(0.01)

        tmp_before = float(compute_tmp(flux / 3600.0, state, params))

        state2, tmp_after, _, _ = mem_step(state, jnp.array(50.0), flux, air, jnp.array(0.0), params, dt)
        assert float(tmp_after) > tmp_before

    def test_backwash_reduces_reversible_fouling(self) -> None:
        params = MembraneParams()
        state = mem_reset(params, jax.random.PRNGKey(1))
        dt = jnp.array(0.01)

        # Accumulate fouling
        for _ in range(50):
            state, _, _, _ = mem_step(state, jnp.array(50.0), jnp.array(0.03), jnp.array(0.0), jnp.array(0.0), params, dt)

        r_rev_before = float(state.r_reversible)
        assert r_rev_before > 0.0

        # Backwash
        state_bw, _, _, _ = mem_step(state, jnp.array(50.0), jnp.array(0.03), jnp.array(0.0), jnp.array(1.0), params, dt)
        assert float(state_bw.r_reversible) < r_rev_before

    def test_air_scour_slows_fouling(self) -> None:
        params = MembraneParams()
        dt = jnp.array(0.01)

        state_no_air = mem_reset(params, jax.random.PRNGKey(2))
        state_air = mem_reset(params, jax.random.PRNGKey(2))

        for _ in range(20):
            state_no_air, _, _, _ = mem_step(state_no_air, jnp.array(50.0), jnp.array(0.03), jnp.array(0.0), jnp.array(0.0), params, dt)
            state_air, _, _, _ = mem_step(state_air, jnp.array(50.0), jnp.array(0.03), jnp.array(1.0), jnp.array(0.0), params, dt)

        assert float(state_air.r_reversible) < float(state_no_air.r_reversible)

    def test_permeate_tss_much_lower_than_feed(self) -> None:
        params = MembraneParams()
        state = mem_reset(params, jax.random.PRNGKey(3))
        _, _, perm_tss, _ = mem_step(state, jnp.array(50.0), jnp.array(0.03), jnp.array(0.5), jnp.array(0.0), params, jnp.array(0.01))
        assert float(perm_tss) < 1.0  # 99.9% rejection

    def test_jit_compatible(self) -> None:
        params = MembraneParams()
        state = mem_reset(params, jax.random.PRNGKey(4))
        jit_step = jax.jit(lambda s: mem_step(s, jnp.array(50.0), jnp.array(0.03), jnp.array(0.5), jnp.array(0.0), params, jnp.array(0.01)))
        s2, tmp, pt, q = jit_step(state)
        assert jnp.isfinite(tmp)


# ---------------------------------------------------------------------------
# Chemistry: Coagulation
# ---------------------------------------------------------------------------
class TestCoagulation:
    def test_optimal_dose_gives_peak_removal(self) -> None:
        params = CoagulationParams()
        feed_tss = jnp.array(100.0)
        turb = jnp.array(10.0)

        _, eta_opt = coagulate(feed_tss, turb, jnp.array(params.dose_opt_base), params)
        _, eta_low = coagulate(feed_tss, turb, jnp.array(params.dose_opt_base * 0.3), params)
        _, eta_high = coagulate(feed_tss, turb, jnp.array(params.dose_opt_base * 3.0), params)

        assert float(eta_opt) > float(eta_low)
        assert float(eta_opt) > float(eta_high)

    def test_effluent_less_than_feed(self) -> None:
        params = CoagulationParams()
        eff, eta = coagulate(jnp.array(100.0), jnp.array(10.0), jnp.array(30.0), params)
        assert float(eff) < 100.0
        assert float(eff) > 0.0

    def test_high_turbidity_shifts_optimal_dose(self) -> None:
        params = CoagulationParams()
        feed_tss = jnp.array(200.0)

        _, eta_low_turb = coagulate(feed_tss, jnp.array(5.0), jnp.array(30.0), params)
        _, eta_high_turb = coagulate(feed_tss, jnp.array(30.0), jnp.array(30.0), params)
        # At 30 mg/L dose and ref turbidity 10, the optimal dose is 30.
        # Higher turbidity shifts optimal dose higher, so 30 mg/L is underdose → less removal
        assert float(eta_high_turb) < float(eta_low_turb)

    def test_jit_compatible(self) -> None:
        params = CoagulationParams()
        jit_coag = jax.jit(lambda d: coagulate(jnp.array(100.0), jnp.array(10.0), d, params))
        eff, eta = jit_coag(jnp.array(30.0))
        assert jnp.isfinite(eff)
        assert jnp.isfinite(eta)


# ---------------------------------------------------------------------------
# Unit model: Anaerobic Digester (ADM1)
# ---------------------------------------------------------------------------
class TestADM1Unit:
    def test_biogas_production_positive(self) -> None:
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(0))
        state2, q_biogas, ch4_frac, ph = adm1_step(
            state, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167),
        )
        assert float(q_biogas) > 0.0

    def test_ch4_fraction_in_range(self) -> None:
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(1))
        _, _, ch4_frac, _ = adm1_step(
            state, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167),
        )
        assert 0.3 < float(ch4_frac) < 0.9

    def test_ph_in_operating_range(self) -> None:
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(2))
        _, _, _, ph = adm1_step(
            state, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167),
        )
        assert 5.0 < float(ph) < 9.0

    def test_vfa_stays_controlled_at_steady_feed(self) -> None:
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(3))
        dt = jnp.array(0.04167)
        for _ in range(100):
            state, _, _, _ = adm1_step(state, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, dt)
        assert float(state.s_vfa) < 2000.0

    def test_jit_compatible(self) -> None:
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(4))
        jit_step = jax.jit(lambda s: adm1_step(s, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167)))
        s2, qb, ch4, ph = jit_step(state)
        assert jnp.isfinite(qb)
        assert jnp.isfinite(ph)


# ---------------------------------------------------------------------------
# Unit model: ASM2d
# ---------------------------------------------------------------------------
class TestASM2dUnit:
    def test_reactions_output_shape(self) -> None:
        params = ASM2dParams()
        state = make_default_influent_asm2d()
        state = state.at[4].set(2500.0).at[5].set(150.0)
        state = state.at[7].set(0.0)  # anaerobic
        state = state.at[14].set(500.0).at[15].set(100.0).at[16].set(100.0)

        dc = reactions_asm2d(state, params)
        assert dc.shape == (N_COMPONENTS_ASM2D,)

    def test_anaerobic_pao_releases_phosphate(self) -> None:
        params = ASM2dParams()
        state = make_default_influent_asm2d()
        state = state.at[4].set(2500.0).at[5].set(150.0)
        state = state.at[7].set(0.0).at[8].set(0.0)  # anaerobic: no O2, no NO3
        state = state.at[14].set(500.0).at[15].set(50.0).at[16].set(150.0)
        state = state.at[13].set(2.0)  # low initial PO4

        dc = reactions_asm2d(state, params)
        # Under anaerobic conditions, PAOs release P → dS_PO4/dt > 0
        assert float(dc[13]) > 0.0

    def test_aerobic_pao_takes_up_phosphate(self) -> None:
        params = ASM2dParams()
        state = make_default_influent_asm2d()
        state = state.at[4].set(2500.0).at[5].set(150.0)
        state = state.at[7].set(4.0)  # aerobic
        state = state.at[14].set(500.0).at[15].set(200.0).at[16].set(50.0)
        state = state.at[13].set(10.0)  # high PO4 available for uptake

        dc = reactions_asm2d(state, params)
        # Under aerobic conditions with PHA and PO4 available, PP storage → dS_PO4/dt < 0
        assert float(dc[13]) < 0.0

    def test_default_influent_correct_size(self) -> None:
        inf = make_default_influent_asm2d()
        assert inf.shape == (N_COMPONENTS_ASM2D,)

    def test_jit_compatible(self) -> None:
        params = ASM2dParams()
        state = make_default_influent_asm2d().at[4].set(2500.0).at[14].set(500.0)
        jit_rxn = jax.jit(lambda s: reactions_asm2d(s, params))
        dc = jit_rxn(state)
        assert jnp.all(jnp.isfinite(dc))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def _run_benchmark_suite(
    make_fn: Callable,
    config: Any,
    action_dim: int,
    obs_dim: int,
    n_stability_steps: int = 100,
):
    """Shared test logic for all benchmarks: shapes, JIT, stability."""
    reset_fn, step_fn = make_fn(config)
    rng = jax.random.PRNGKey(42)

    # Reset
    state, obs = reset_fn(rng)
    assert obs.shape == (obs_dim,), f"obs shape {obs.shape} != ({obs_dim},)"

    # Step
    action = jnp.ones(action_dim) * 0.5
    rng, k = jax.random.split(rng)
    state2, obs2, reward, done, info = step_fn(state, action, k)
    assert obs2.shape == (obs_dim,)
    assert reward.shape == ()
    assert jnp.isfinite(reward)

    # JIT compile
    jit_step = jax.jit(step_fn)
    rng, k = jax.random.split(rng)
    state3, obs3, reward3, done3, info3 = jit_step(state, action, k)
    assert obs3.shape == (obs_dim,)
    assert jnp.isfinite(reward3)

    # Multi-step stability
    rng = jax.random.PRNGKey(99)
    state_s, obs_s = reset_fn(rng)
    reward_s = jnp.array(0.0)
    for _ in range(n_stability_steps):
        rng, k = jax.random.split(rng)
        state_s, obs_s, reward_s, _, _ = step_fn(state_s, action, k)
    assert jnp.all(jnp.isfinite(obs_s)), f"NaN in obs after {n_stability_steps} steps"
    assert jnp.isfinite(reward_s)


# ---------------------------------------------------------------------------
# Benchmark: Primary Clarifier (1D action, 5D obs)
# ---------------------------------------------------------------------------
class TestPrimaryClarifierBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_primary_clarifier_benchmark, PrimaryClarifierConfig(), action_dim=1, obs_dim=5)


# ---------------------------------------------------------------------------
# Benchmark: Dewatering (2D action, 5D obs)
# ---------------------------------------------------------------------------
class TestDewateringBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_dewatering_benchmark, DewateringConfig(), action_dim=2, obs_dim=5)


# ---------------------------------------------------------------------------
# Benchmark: Membrane Fouling (3D action, 6D obs)
# ---------------------------------------------------------------------------
class TestMembraneFoulingBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_membrane_fouling_benchmark, MembraneFoulingConfig(), action_dim=3, obs_dim=6)


# ---------------------------------------------------------------------------
# Benchmark: Anaerobic Digester (2D action, 7D obs)
# ---------------------------------------------------------------------------
class TestAnaerobicDigesterBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_anaerobic_digester_benchmark, AnaerobicDigesterConfig(), action_dim=2, obs_dim=7)


# ---------------------------------------------------------------------------
# Benchmark: Reject Water (2D action, 6D obs)
# ---------------------------------------------------------------------------
class TestRejectWaterBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_reject_water_benchmark, RejectWaterConfig(), action_dim=2, obs_dim=6)


# ---------------------------------------------------------------------------
# Benchmark: Drinking Water Train (3D action, 8D obs)
# ---------------------------------------------------------------------------
class TestDrinkingWaterTrainBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_drinking_water_train_benchmark, DrinkingWaterTrainConfig(), action_dim=3, obs_dim=8)


# ---------------------------------------------------------------------------
# Benchmark: Bio-P (2D action, 7D obs)
# ---------------------------------------------------------------------------
class TestBioPBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_bio_p_benchmark, BioPConfig(), action_dim=2, obs_dim=7)


# ---------------------------------------------------------------------------
# Benchmark: Combined N+P (4D action, 9D obs)
# ---------------------------------------------------------------------------
class TestCombinedNPBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_combined_np_benchmark, CombinedNPConfig(), action_dim=4, obs_dim=9)


# ---------------------------------------------------------------------------
# Benchmark: BSM1 Takacs (3D action, 12D obs)
# ---------------------------------------------------------------------------
class TestBSM1TakacsBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_bsm1_takacs_benchmark, BSM1TakacsConfig(), action_dim=3, obs_dim=12)
