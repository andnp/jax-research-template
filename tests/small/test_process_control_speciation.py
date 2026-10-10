"""Ideal charge-balance speciation contracts, checked against closed-form oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY
from process_control.chemistry.speciation import (
    SolveControls,
    SpeciationSystem,
    evaluate_constants,
    fractions,
    make_speciation_system,
    solve_ideal_charge_balance,
)

T25 = jnp.float32(298.15)
C0 = 1000.0


def _system(*families: str):
    return make_speciation_system(DEFAULT_REGISTRY, families)


def _solve(
    system: SpeciationSystem,
    totals: jax.Array,
    strong_charge: float | jax.Array = 0.0,
    temperature_k: jax.Array = T25,
    controls: SolveControls | None = None,
):
    constants = evaluate_constants(system, temperature_k)
    totals = jnp.asarray(totals, dtype=jnp.float32)
    strong = jnp.asarray(strong_charge, dtype=jnp.float32)
    if controls is None:
        return solve_ideal_charge_balance(system, totals, strong, constants)
    return solve_ideal_charge_balance(system, totals, strong, constants, controls)


def _water_ph_oracle(strong_charge_eq_per_m3: float, log_kw: float):
    """pH of water with net strong-ion charge s: h^2 + (s/C0) h - Kw = 0, in a cancellation-safe form."""
    kw = 10.0**log_kw
    s = strong_charge_eq_per_m3 / C0
    root = np.sqrt(s * s + 4.0 * kw)
    h = kw / ((s + root) / 2.0) if s >= 0.0 else (-s + root) / 2.0
    return -np.log10(h)


def _log_kw(system: SpeciationSystem, temperature_k: jax.Array = T25):
    return float(evaluate_constants(system, temperature_k).log_kw)


class TestFractions:
    def test_monoprotic_fractions_at_and_around_the_pka(self):
        """Half dissociated at pH = pKa; a unit away gives 10:1 and 1:10."""
        log_ka = jnp.array([-9.25])
        assert jnp.allclose(fractions(log_ka, jnp.float32(9.25)), jnp.array([0.5, 0.5]), atol=1e-6)
        assert jnp.allclose(fractions(log_ka, jnp.float32(8.25)), jnp.array([10 / 11, 1 / 11]), atol=1e-6)
        assert jnp.allclose(fractions(log_ka, jnp.float32(10.25)), jnp.array([1 / 11, 10 / 11]), atol=1e-6)

    @pytest.mark.parametrize("ph", [-2.0, 3.0, 6.35, 8.3, 10.33, 12.0, 16.0])
    def test_diprotic_fractions_match_the_closed_form(self, ph: float):
        """Carbonate-like ladder agrees with h^2, K1 h, K1 K2 over their sum, including extreme pH."""
        k1, k2 = 10**-6.35, 10**-10.33
        h = 10.0**-ph
        expected = np.array([h * h, k1 * h, k1 * k2]) / (h * h + k1 * h + k1 * k2)
        got = fractions(jnp.array([-6.35, -10.33]), jnp.float32(ph))
        assert np.allclose(got, expected, atol=1e-5)
        assert bool(jnp.all(jnp.isfinite(got))) and float(jnp.sum(got)) == pytest.approx(1.0, abs=1e-5)

    def test_batched_ph_gives_one_row_per_ph(self):
        """Fractions broadcast over a pH batch along a trailing form axis."""
        out = fractions(jnp.array([-6.35, -10.33]), jnp.linspace(0.0, 14.0, 5))
        assert out.shape == (5, 3)


class TestWaterAndStrongIons:
    def test_pure_water_is_neutral_at_half_pkw(self):
        """With no families and no strong ions, pH = pKw / 2."""
        system = _system()
        result = _solve(system, jnp.zeros((0,)))
        assert bool(result.converged)
        assert float(result.ph) == pytest.approx(-_log_kw(system) / 2.0, abs=2e-3)
        assert result.forms_mol_per_m3.shape == (0,)

    @pytest.mark.parametrize("strong_charge", [-100.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 100.0])
    def test_strong_acid_or_base_matches_the_quadratic(self, strong_charge: float):
        """Net strong-ion charge in eq/m3 shifts pH as the independent quadratic predicts (positive means base)."""
        system = _system()
        result = _solve(system, jnp.zeros((0,)), strong_charge)
        assert bool(result.converged)
        assert float(result.ph) == pytest.approx(_water_ph_oracle(strong_charge, _log_kw(system)), abs=3e-3)

    def test_unit_conversion_is_per_cubic_metre(self):
        """1 eq/m3 of strong acid is 1 mM: pH 3, which exposes a factor-1000 unit slip."""
        result = _solve(_system(), jnp.zeros((0,)), -1.0)
        assert float(result.ph) == pytest.approx(3.0, abs=1e-3)


class TestWeakFamilies:
    def test_weak_acid_matches_the_cubic_oracle(self):
        """HOCl alone satisfies h^3 + Ka h^2 - (Kw + C Ka) h - Kw Ka = 0 for its H+ concentration."""
        system = _system("hypochlorous")
        constants = evaluate_constants(system, T25)
        ka, kw = 10.0 ** float(constants.log_ka[0]), 10.0 ** float(constants.log_kw)
        total = 5.0
        roots = np.roots([1.0, ka, -(kw + total / C0 * ka), -kw * ka])
        h = max(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0)
        result = _solve(system, jnp.array([total]))
        assert float(result.ph) == pytest.approx(-np.log10(h), abs=3e-3)

    def test_forms_sum_to_family_totals_and_balance_charge(self):
        """Flattened forms conserve each total, and the reconstructed charge sums to zero."""
        system = _system("carbonate", "ammonium", "sulfide")
        totals = jnp.array([4.0, 2.0, 0.5])
        result = _solve(system, totals, 1.5)

        forms = np.asarray(result.forms_mol_per_m3)
        sizes = np.diff(system.form_offsets)
        sums = [forms[a : a + n].sum() for a, n in zip(system.form_offsets, sizes, strict=False)]
        assert np.allclose(sums, np.asarray(totals), rtol=1e-4)

        ph = float(result.ph)
        charge = 1.5 + C0 * 10.0**-ph - C0 * 10.0 ** (_log_kw(system) + ph) + forms @ np.array(system.form_charges)
        assert abs(charge) < 1e-3

    def test_weak_base_raises_ph_and_added_acid_lowers_it(self):
        """Ammonia buffers above neutral; strong acid on carbonate pushes pH below neutral."""
        base = _solve(_system("ammonium"), jnp.array([1.0]))
        acid = _solve(_system("carbonate"), jnp.array([1.0]), -0.5)
        assert float(base.ph) > 7.0 > float(acid.ph)

    def test_zero_totals_are_valid(self):
        """All-zero totals reduce to pure water with all-zero forms."""
        system = _system("carbonate", "ammonium")
        result = _solve(system, jnp.zeros(2))
        assert bool(result.converged)
        assert np.allclose(result.forms_mol_per_m3, 0.0)
        assert float(result.ph) == pytest.approx(-_log_kw(system) / 2.0, abs=2e-3)

    def test_zero_total_family_does_not_change_the_solution(self):
        """Adding a family with zero total leaves pH unchanged."""
        base = _solve(_system("carbonate"), jnp.array([3.0]), 1.0)
        extended = _solve(_system("carbonate", "sulfide"), jnp.array([3.0, 0.0]), 1.0)
        assert float(extended.ph) == pytest.approx(float(base.ph), abs=1e-4)

    def test_reordering_families_permutes_outputs_only(self):
        """Swapping family order and totals gives the same pH and the permuted form blocks."""
        a, b = _system("carbonate", "ammonium"), _system("ammonium", "carbonate")
        ra = _solve(a, jnp.array([4.0, 2.0]), 0.5)
        rb = _solve(b, jnp.array([2.0, 4.0]), 0.5)
        assert float(ra.ph) == pytest.approx(float(rb.ph), abs=1e-4)
        assert np.allclose(ra.forms_mol_per_m3[:3], rb.forms_mol_per_m3[2:], rtol=1e-3, atol=1e-5)
        assert np.allclose(ra.forms_mol_per_m3[3:], rb.forms_mol_per_m3[:2], rtol=1e-3, atol=1e-5)

    def test_overlapping_families_are_rejected(self):
        """A form cannot be counted in two independent totals."""
        with pytest.raises(ValueError, match="overlap"):
            _system("carbonate", "carbonate")

    def test_totals_axis_must_match_family_count(self):
        """A totals vector of the wrong length fails loudly instead of broadcasting."""
        system = _system("carbonate", "ammonium")
        with pytest.raises(ValueError, match="families"):
            solve_ideal_charge_balance(
                system, jnp.ones(3), jnp.float32(0.0), evaluate_constants(system, T25)
            )


class TestStatus:
    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_invalid_totals_are_flagged_not_clipped(self, bad: float):
        """Negative or non-finite totals give an invalid, non-converged result with NaN pH."""
        result = _solve(_system("carbonate"), jnp.array([bad]))
        assert not bool(result.inputs_valid) and not bool(result.converged)
        assert bool(jnp.isnan(result.ph))

    def test_non_finite_strong_charge_is_invalid(self):
        """A NaN strong-ion charge is rejected."""
        result = _solve(_system("carbonate"), jnp.array([1.0]), float("nan"))
        assert not bool(result.inputs_valid) and not bool(result.converged)

    def test_invalid_lane_does_not_affect_valid_lanes(self):
        """In a batch, a bad lane leaves its neighbours' solutions intact."""
        system = _system("carbonate")
        totals = jnp.array([[3.0], [-1.0], [3.0]])
        result = _solve(system, totals, jnp.array([0.0, 0.0, 0.0]))
        assert result.converged.tolist() == [True, False, True]
        assert float(result.ph[0]) == pytest.approx(float(result.ph[2]))

    def test_invalid_temperature_is_flagged(self):
        """A non-positive temperature makes the constants, hence the solve, invalid."""
        result = _solve(_system("ammonium"), jnp.array([1.0]), temperature_k=jnp.float32(-5.0))
        assert not bool(result.inputs_valid) and not bool(result.converged)

    def test_converged_and_in_domain_are_distinct(self):
        """Extrapolated temperatures still converge but are not in-domain; a starved solver is the reverse."""
        system = _system("hypochlorous")
        hot = _solve(system, jnp.array([1.0]), temperature_k=jnp.float32(340.0))
        starved = _solve(system, jnp.array([1.0]), 2.0, controls=SolveControls(max_iterations=0))
        assert bool(hot.converged) and not bool(hot.constants_in_domain)
        assert bool(starved.constants_in_domain) and not bool(starved.converged)

    def test_unbracketable_problem_reports_bracket_failure(self):
        """If the search bounds miss the root, bracket_ok is False and convergence is not claimed."""
        result = _solve(_system(), jnp.zeros((0,)), controls=SolveControls(ph_bounds=(8.0, 9.0)))
        assert not bool(result.bracket_ok) and not bool(result.converged)

    def test_sulfide_flags_model_ph_above_its_ceiling(self):
        """A strongly basic sulfide solution leaves the S--free model's valid pH range."""
        system = _system("sulfide")
        mild = _solve(system, jnp.array([1.0]), 0.0)
        basic = _solve(system, jnp.array([1.0]), 50.0)
        assert bool(mild.model_ph_in_range) and not bool(basic.model_ph_in_range)


class TestExecutionModes:
    def test_eager_jit_and_vmap_agree(self):
        """The same problem solved eagerly, jitted and vmapped gives the same pH and status."""
        system = _system("carbonate", "ammonium")
        temps = jnp.array([283.15, 298.15, 308.15])
        totals = jnp.array([[4.0, 2.0], [1.0, 0.1], [10.0, 5.0]])
        strong = jnp.array([0.5, -0.2, 2.0])

        def solve(totals: jax.Array, strong: jax.Array, t: jax.Array):
            return solve_ideal_charge_balance(system, totals, strong, evaluate_constants(system, t))

        batched = solve(totals, strong, temps)
        mapped = jax.vmap(solve)(totals, strong, temps)
        compiled = jax.jit(solve)(totals, strong, temps)
        scalar = [solve(totals[i], strong[i], temps[i]).ph for i in range(3)]

        assert batched.ph.shape == (3,) and batched.forms_mol_per_m3.shape == (3, 5)
        assert bool(jnp.all(batched.converged))
        for other in (mapped, compiled):
            assert np.allclose(batched.ph, other.ph, atol=1e-4)
            assert np.allclose(batched.forms_mol_per_m3, other.forms_mol_per_m3, rtol=1e-3, atol=1e-5)
            assert other.converged.tolist() == batched.converged.tolist()
        assert np.allclose(batched.ph, jnp.stack(scalar), atol=1e-4)
