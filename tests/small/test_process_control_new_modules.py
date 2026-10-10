from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.anaerobic_digester import (
    AnaerobicDigesterConfig,
    make_anaerobic_digester_benchmark,
)
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
from process_control.benchmarks.reject_water import RejectWaterBenchState, RejectWaterConfig, make_reject_water_benchmark
from process_control.chemistry.coagulation import CoagulationParams, coagulate
from process_control.units.anaerobic_digester import ADM1Params, ADM1State
from process_control.units.anaerobic_digester import reset as adm1_reset
from process_control.units.anaerobic_digester import step as adm1_step
from process_control.units.asm2d import N_COMPONENTS_ASM2D, ASM2dParams, make_default_influent_asm2d, reactions_asm2d
from process_control.units.dewatering import DewateringParams, DewateringState
from process_control.units.dewatering import reset as dw_reset
from process_control.units.dewatering import step as dw_step
from process_control.units.membrane import MembraneParams, MembraneState, compute_tmp
from process_control.units.membrane import reset as mem_reset
from process_control.units.membrane import step as mem_step
from process_control.units.primary_clarifier import (
    PrimaryClarifierParams,
    PrimaryClarifierState,
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
        state2, eff_tss, und_tss, _ = pc_step(
            state, jnp.array(250.0), jnp.array(600.0), jnp.array(20.0), params, jnp.array(0.02),
        )
        assert float(eff_tss) < 250.0
        assert float(eff_tss) > 0.0

    def test_underflow_tss_exceeds_effluent(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(1))
        _, eff_tss, und_tss, _ = pc_step(
            state, jnp.array(250.0), jnp.array(600.0), jnp.array(20.0), params, jnp.array(0.02),
        )
        assert float(und_tss) > float(eff_tss)

    def test_sludge_mass_increases_with_no_wasting(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(2))
        state2, _, _, _ = pc_step(
            state, jnp.array(250.0), jnp.array(600.0), jnp.array(0.0), params, jnp.array(0.02),
        )
        assert float(state2.sludge_mass) >= float(state.sludge_mass)

    def test_sludge_overflow_is_reported(self) -> None:
        """Clarifier capacity clipping reports positive and zero overflow."""
        params = PrimaryClarifierParams(max_sludge_mass=1.0)
        state = PrimaryClarifierState(sludge_mass=jnp.array(0.9))
        _, _, _, overflow = pc_step(
            state, jnp.array(1e6), jnp.array(500.0), jnp.array(0.0), params, jnp.array(1.0),
        )
        assert overflow > 0.0

    def test_jit_compatible(self) -> None:
        params = PrimaryClarifierParams()
        state = pc_reset(params, jax.random.PRNGKey(3))
        def step_fn(s: PrimaryClarifierState):
            return pc_step(s, jnp.array(250.0), jnp.array(600.0), jnp.array(20.0), params, jnp.array(0.02))

        jit_step = jax.jit(step_fn)
        state2, eff, und, _ = jit_step(state)
        assert jnp.isfinite(eff)
        assert jnp.isfinite(und)

    def test_wastage_never_exceeds_available_mass(self) -> None:
        """A huge waste rate can't drive sludge_mass negative (no manufactured mass)."""
        params = PrimaryClarifierParams()
        state = PrimaryClarifierState(sludge_mass=jnp.array(100.0))
        state2, _, _, _ = pc_step(
            state, jnp.array(0.0), jnp.array(600.0), jnp.array(1e6), params, jnp.array(1.0),
        )
        assert float(state2.sludge_mass) >= 0.0

    def test_overflow_routes_to_effluent_not_deleted(self) -> None:
        """Overflow beyond capacity raises effluent TSS instead of vanishing."""
        capped_params = PrimaryClarifierParams(max_sludge_mass=1.0)
        uncapped_params = PrimaryClarifierParams(max_sludge_mass=1e12)
        state = PrimaryClarifierState(sludge_mass=jnp.array(0.9))
        _, eff_tss_capped, _, overflow = pc_step(
            state, jnp.array(250.0), jnp.array(500.0), jnp.array(0.0), capped_params, jnp.array(1.0),
        )
        _, eff_tss_uncapped, _, no_overflow = pc_step(
            state, jnp.array(250.0), jnp.array(500.0), jnp.array(0.0), uncapped_params, jnp.array(1.0),
        )
        assert float(overflow) > 0.0
        assert float(no_overflow) == 0.0
        assert float(eff_tss_capped) > float(eff_tss_uncapped)


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

        _, dry_lo, filt_lo, _, _ = dw_step(state, feed_tss, q_feed, jnp.array(0.0), jnp.array(0.5), params, dt)
        _, dry_hi, filt_hi, _, _ = dw_step(state, feed_tss, q_feed, jnp.array(10.0), jnp.array(0.5), params, dt)

        assert float(dry_hi) > float(dry_lo)
        assert float(filt_hi) < float(filt_lo)

    def test_outputs_positive(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(1))
        state2, dryness, filt_tss, q_filt, _ = dw_step(
            state, jnp.array(10000.0), jnp.array(30.0), jnp.array(5.0), jnp.array(0.5), params, jnp.array(0.05),
        )
        assert float(dryness) > 0.0
        assert float(filt_tss) >= 0.0
        assert float(q_filt) >= 0.0

    def test_cumulative_cake_increases(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(2))
        state2, _, _, _, _ = dw_step(
            state, jnp.array(10000.0), jnp.array(30.0), jnp.array(5.0), jnp.array(0.5), params, jnp.array(0.05),
        )
        assert float(state2.cake_produced) > 0.0

    def test_jit_compatible(self) -> None:
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(3))
        def step_fn(s: DewateringState):
            return dw_step(s, jnp.array(10000.0), jnp.array(30.0), jnp.array(5.0), jnp.array(0.5), params, jnp.array(0.05))

        jit_step = jax.jit(step_fn)
        s2, dry, filt, q, _ = jit_step(state)
        assert jnp.isfinite(dry)

    def test_water_and_solids_mass_balance(self) -> None:
        """Cake + filtrate flow must reconstruct the actual throughput, and
        captured + lost solids must reconstruct solids fed in — catches the
        cake-volume formula silently destroying or creating mass."""
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(0))
        feed_tss = jnp.array(10000.0)
        q_feed = jnp.array(30.0)
        belt_speed = jnp.array(0.8)
        dt = jnp.array(0.05)

        new_state, dryness, filtrate_tss, q_filtrate, _ = dw_step(
            state, feed_tss, q_feed, jnp.array(5.0), belt_speed, params, dt,
        )

        q_actual = jnp.minimum(q_feed, belt_speed * params.max_throughput)
        cake_mass_flow = (new_state.cake_produced - state.cake_produced) / dt  # kg/h
        implied_q_cake = cake_mass_flow / (dryness * 1000.0)  # kg/h / (kg/m3)
        assert jnp.allclose(implied_q_cake + q_filtrate, q_actual, rtol=1e-5)

        solids_in = feed_tss * q_actual * dt / 1e3
        solids_lost = filtrate_tss * q_filtrate * dt / 1e3
        assert jnp.allclose((new_state.cake_produced - state.cake_produced) + solids_lost, solids_in, rtol=1e-5)


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
        def step_fn(s: MembraneState):
            return mem_step(s, jnp.array(50.0), jnp.array(0.03), jnp.array(0.5), jnp.array(0.0), params, jnp.array(0.01))

        jit_step = jax.jit(step_fn)
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
        def coagulate_fn(d: jax.Array):
            return coagulate(jnp.array(100.0), jnp.array(10.0), d, params)

        jit_coag = jax.jit(coagulate_fn)
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
        def step_fn(s: ADM1State):
            return adm1_step(s, jnp.array(30000.0), jnp.array(150.0), jnp.array(35.0), params, jnp.array(0.04167))

        jit_step = jax.jit(step_fn)
        s2, qb, ch4, ph = jit_step(state)
        assert jnp.isfinite(qb)
        assert jnp.isfinite(ph)


def _adm1_cod_pool(s: ADM1State) -> jax.Array:
    # CO2 carries no COD; every other tracked pool is mg-COD/L.
    return s.x_substrate + s.x_biomass + s.x_inert + s.s_vfa + s.s_ch4


class TestADM1CODConservation:
    def test_cod_balance_over_multiple_steps(self) -> None:
        """Input COD equals effluent + CH4-COD vented + storage change, within numerical tolerance."""
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(5))
        dt = jnp.array(0.04167)
        feed_cod = jnp.array(30000.0)
        q_feed = jnp.array(125.0)
        temperature = jnp.array(35.0)
        dilution = q_feed / params.volume

        pool_before = _adm1_cod_pool(state)

        def body(state: ADM1State, _step: None):
            pool_start = _adm1_cod_pool(state)
            state, q_biogas, ch4_frac, _ = adm1_step(state, feed_cod, q_feed, temperature, params, dt)
            ch4_vol = q_biogas * ch4_frac * dt
            return state, (dilution * (feed_cod - pool_start) * dt, ch4_vol * 1000.0 / 22.4 * 64.0 / params.volume)

        state, (net_dilution, ch4_cod_out) = jax.lax.scan(body, state, None, length=200)
        net_dilution_total = net_dilution.sum()
        ch4_cod_out_total = ch4_cod_out.sum()

        delta_storage = _adm1_cod_pool(state) - pool_before
        residual = net_dilution_total - ch4_cod_out_total - delta_storage
        assert abs(float(residual)) < 0.01 * abs(float(net_dilution_total))

    def test_organic_overload_drops_ph(self) -> None:
        """A hydraulic/organic overload accumulates VFA and drops pH (acid-crash failure mode)."""
        params = ADM1Params()
        state = adm1_reset(30000.0, params, jax.random.PRNGKey(6))
        dt = jnp.array(0.04167)
        feed_cod = jnp.array(30000.0)
        temperature = jnp.array(35.0)

        def run(state: ADM1State, q_feed: float, n_steps: int):
            def body(state: ADM1State, _step: None):
                state, _, _, ph = adm1_step(state, feed_cod, jnp.array(q_feed), temperature, params, dt)
                return state, ph

            return jax.lax.scan(body, state, None, length=n_steps)

        state, baseline_ph = run(state, 125.0, 24 * 10)
        ph_baseline = baseline_ph[-1]

        # dilution rate far exceeds mu_max -> washout + VFA accumulation
        _, overload_ph = run(state, 1500.0, 120)
        ph_overload = overload_ph[-1]

        assert float(ph_overload) < float(ph_baseline) - 0.5


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
        def reactions_fn(s: jax.Array):
            return reactions_asm2d(s, params)

        jit_rxn = jax.jit(reactions_fn)
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
# Benchmark: Membrane Fouling (3D action, 7D obs)
# ---------------------------------------------------------------------------
class TestMembraneFoulingBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_membrane_fouling_benchmark, MembraneFoulingConfig(), action_dim=3, obs_dim=7)


# ---------------------------------------------------------------------------
# Benchmark: Anaerobic Digester (2D action, 7D obs)
# ---------------------------------------------------------------------------
class TestAnaerobicDigesterBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_anaerobic_digester_benchmark, AnaerobicDigesterConfig(), action_dim=2, obs_dim=7)


# ---------------------------------------------------------------------------
# Benchmark: Reject Water (1D action, 6D obs)
# ---------------------------------------------------------------------------
class TestRejectWaterBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_reject_water_benchmark, RejectWaterConfig(), action_dim=1, obs_dim=6)

    def test_tank_overflow_is_reported(self) -> None:
        """Reject tank capacity clipping reports positive and zero overflow."""
        reset_fn, step_fn = make_reject_water_benchmark(RejectWaterConfig(reject_tank_max=1.0))
        state, _ = reset_fn(jax.random.PRNGKey(1))
        for index in range(10):
            state, _, _, _, info = step_fn(state, jnp.array([0.0]), jax.random.PRNGKey(index))
        assert info["overflow"] > 0.0

        _, _, _, _, drained_info = step_fn(
            state, jnp.array([1.0]), jax.random.PRNGKey(99),
        )
        assert drained_info["overflow"] == 0.0

    def test_overflow_is_priced_as_lost_production_steps(self) -> None:
        """Overflow is charged as the production steps it discards, with NH₄ kept under the limit.

        A shut valve on a nearly full tank drops one step of production; the reward must
        carry exactly that many steps' worth of penalty, so tank size does not change the price.
        """
        config = RejectWaterConfig(reject_tank_max=0.1)
        reset_fn, step_fn = make_reject_water_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        _, _, reward, _, info = step_fn(state, jnp.array([0.0]), jax.random.PRNGKey(1))

        assert float(info["overflow"]) > 0.0
        assert float(info["main_nh4"]) < config.nh4_limit
        lost_steps = info["overflow"] / (config.reject_production * config.dt)
        assert jnp.allclose(reward, -config.reward_w_overflow * lost_steps)

    def test_reward_scores_true_nh4_not_sensor_reading(self) -> None:
        """Reward matches the true NH₄ and overflow even when the sensed NH₄ is dominated by noise."""
        config = RejectWaterConfig(nh4_baseline=10.0, nh4_noise_std=20.0)
        reset_fn, step_fn = make_reject_water_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        _, _, reward, _, info = step_fn(state, jnp.array([0.0]), jax.random.PRNGKey(1))

        violation = jnp.maximum(info["main_nh4"] - config.nh4_limit, 0.0)
        assert float(violation) > 0.0
        expected = -(
            config.reward_w_nh4 * (violation / config.nh4_limit) ** 2
            + config.reward_w_overflow * info["overflow"] / (config.reject_production * config.dt)
        )
        assert jnp.allclose(reward, expected)

    def test_observed_tank_level_matches_post_step_tank(self) -> None:
        """Observed tank level is the post-step fraction, so a shut valve shows exactly one production increment."""
        config = RejectWaterConfig()
        reset_fn, step_fn = make_reject_water_benchmark(config)
        state, obs = reset_fn(jax.random.PRNGKey(0))
        assert float(obs[1]) == 0.5

        _, obs2, _, _, info = step_fn(state, jnp.array([0.0]), jax.random.PRNGKey(1))
        assert jnp.allclose(obs2[1], info["tank_volume"] / config.reject_tank_max)
        assert jnp.allclose(obs2[1], 0.5 + config.reject_production * config.dt / config.reject_tank_max)

    def test_nitrification_compliant_baseline_but_reject_surge_violates(self) -> None:
        """Monod nitrification must clear baseline load below the alarm, but
        a sustained reject-water return must still be able to spike past it
        — a constant-rate sink either stays permanently over alarm or never
        responds to load at all."""
        reset_fn, step_fn = make_reject_water_benchmark(RejectWaterConfig(reject_nh4=800.0))
        step_fn = jax.jit(step_fn)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        for index in range(2000):
            state, _, _, _, info = step_fn(state, jnp.array([0.0]), jax.random.PRNGKey(index))
        assert float(info["main_nh4"]) < RejectWaterConfig.nh4_limit

        peak = 0.0
        for index in range(300):
            state, _, _, _, info = step_fn(state, jnp.array([1.0]), jax.random.PRNGKey(3000 + index))
            peak = max(peak, float(info["main_nh4"]))
        assert peak > RejectWaterConfig.nh4_limit


def _constant_return_policy(rate: float) -> Callable[[jax.Array, jax.Array], jax.Array]:
    def policy(obs: jax.Array, key: jax.Array) -> jax.Array:
        return jnp.array([rate])

    return policy


def _headroom_return_policy(config: RejectWaterConfig, cap: float) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Test-only feedforward rule: return what the nitrifying zone can absorb after main-flow load, reading only the observation."""
    pass_through = config.reject_production / config.q_reject_max

    def policy(obs: jax.Array, key: jax.Array) -> jax.Array:
        level, flow_ratio = obs[1], obs[3]
        main_load = flow_ratio * config.mean_flow * config.influent_nh4 / config.plant_volume
        rate = (cap - main_load) * config.plant_volume / (config.reject_nh4 * config.q_reject_max)
        rate = jnp.where(level > 0.85, jnp.maximum(rate, pass_through), rate)
        return jnp.array([jnp.clip(rate, 0.0, 1.0)])

    return policy


def _reject_water_return_per_day(
    policy: Callable[[jax.Array, jax.Array], jax.Array], warm_steps: int, days: int = 3, seeds: int = 4
) -> tuple[float, float]:
    """Mean reward and overflow per day, with warm-up steps excluded from scoring."""
    config = RejectWaterConfig()
    reset_fn, step_fn = make_reject_water_benchmark(config)
    horizon = warm_steps + config.steps_per_day * days

    def episode(key: jax.Array):
        reset_key, run_key = jax.random.split(key)
        state, obs = reset_fn(reset_key)

        def body(
            carry: tuple[RejectWaterBenchState, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[RejectWaterBenchState, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            state, obs, key = carry
            key, action_key, step_key = jax.random.split(key, 3)
            state, obs, reward, _done, info = step_fn(state, policy(obs, action_key), step_key)
            return (state, obs, key), (reward, info["overflow"])

        _, (rewards, overflow) = jax.lax.scan(body, (state, obs, run_key), None, length=horizon)
        return rewards[warm_steps:].sum() / days, overflow[warm_steps:].sum() / days

    returns, overflows = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), seeds))
    return float(returns.mean()), float(overflows.mean())


class TestRejectWaterTimingTradeoff:
    def test_action_sets_return_rate_between_endpoints(self) -> None:
        """Action 0 holds reject water and action 1 returns at the full rate while the reset tank holds enough water."""
        config = RejectWaterConfig()
        reset_fn, step_fn = make_reject_water_benchmark(config)
        state, _ = reset_fn(jax.random.PRNGKey(0))
        _, _, _, _, shut = step_fn(state, jnp.array([0.0]), jax.random.PRNGKey(1))
        _, _, _, _, full = step_fn(state, jnp.array([1.0]), jax.random.PRNGKey(1))
        assert float(shut["q_reject"]) == 0.0
        assert float(full["q_reject"]) == config.q_reject_max

    @pytest.mark.parametrize("days", [3, 7], ids=["three_days", "seven_days"])
    @pytest.mark.parametrize("warm_steps", [0, 300], ids=["reset", "quarter_day_warm_start"])
    def test_policy_ranking_holds_across_start_phases(self, warm_steps: int, days: int) -> None:
        """Shut loses most, pass-through beats full-open, and the headroom rule beats pass-through with no overflow."""
        config = RejectWaterConfig()
        shut, _ = _reject_water_return_per_day(_constant_return_policy(0.0), warm_steps, days)
        pass_through, _ = _reject_water_return_per_day(
            _constant_return_policy(config.reject_production / config.q_reject_max), warm_steps, days
        )
        full_open, _ = _reject_water_return_per_day(_constant_return_policy(1.0), warm_steps, days)
        headroom, headroom_overflow = _reject_water_return_per_day(
            _headroom_return_policy(config, cap=8.0), warm_steps, days
        )
        assert shut < pass_through
        assert full_open < pass_through
        assert pass_through < headroom
        assert headroom_overflow == 0.0

# ---------------------------------------------------------------------------
# Benchmark: Drinking Water Train (3D action, 8D obs)
# ---------------------------------------------------------------------------
class TestDrinkingWaterTrainBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_drinking_water_train_benchmark, DrinkingWaterTrainConfig(), action_dim=3, obs_dim=8)


# ---------------------------------------------------------------------------
# Benchmark: Combined N+P (4D action, 9D obs)
# ---------------------------------------------------------------------------
class TestCombinedNPBenchmark:
    def test_shapes_jit_stability(self) -> None:
        _run_benchmark_suite(make_combined_np_benchmark, CombinedNPConfig(), action_dim=4, obs_dim=9)

