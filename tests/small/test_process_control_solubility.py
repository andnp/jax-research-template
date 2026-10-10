"""Henry's-law solubility evaluation and its forward and inverse relations."""

import jax
import jax.numpy as jnp
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY
from process_control.chemistry.solubility import (
    equilibrium_concentration,
    equilibrium_partial_pressure,
    henry_hcp,
)

SPECIES = ("H2S", "CO2", "O2", "NH3", "CH4")


class TestHenryEvaluation:
    @pytest.mark.parametrize("species", SPECIES)
    def test_reference_temperature_returns_the_tabulated_hcp(self, species: str):
        """At 298.15 K the evaluation is the assessed reference value, in-domain."""
        record = DEFAULT_REGISTRY.get_henry(species)
        result = henry_hcp(record, jnp.float32(298.15))
        assert jnp.allclose(result.value, record.hcp_ref_mol_per_m3_pa, rtol=1e-5)
        assert bool(result.numerically_valid) and bool(result.in_domain)

    @pytest.mark.parametrize("species", SPECIES)
    def test_gas_solubility_falls_as_temperature_rises(self, species: str):
        """These gases dissolve exothermically, so Hcp decreases monotonically with temperature."""
        hcp = henry_hcp(DEFAULT_REGISTRY.get_henry(species), jnp.linspace(280.0, 306.0, 6)).value
        assert bool(jnp.all(jnp.diff(hcp) < 0.0))

    def test_temperature_dependence_follows_the_van_t_hoff_slope(self):
        """ln(Hcp/Hcp_ref) = B (1/T - 1/Tref) for CO2 at 10 C."""
        record = DEFAULT_REGISTRY.get_henry("CO2")
        result = henry_hcp(record, jnp.float32(283.15))
        expected = 3.4e-4 * jnp.exp(2300.0 * (1 / 283.15 - 1 / 298.15))
        assert jnp.allclose(result.value, expected, rtol=1e-4)
        assert float(result.value) > 3.4e-4

    @pytest.mark.parametrize(("temperature_k", "expected"), [(270.0, False), (278.15, True), (308.15, True), (320.0, False)])
    def test_guard_range_controls_in_domain_without_clamping(self, temperature_k: float, expected: bool):
        """Outside the design guard the value is still finite and unclamped but not in-domain."""
        record = DEFAULT_REGISTRY.get_henry("H2S")
        result = henry_hcp(record, jnp.float32(temperature_k))
        assert bool(result.in_domain) is expected
        assert bool(result.numerically_valid)
        assert jnp.allclose(
            result.value,
            record.hcp_ref_mol_per_m3_pa * jnp.exp(record.b_kelvin * (1 / temperature_k - 1 / 298.15)),
            rtol=1e-4,
        )

    @pytest.mark.parametrize("bad_t", [0.0, -5.0, float("nan")])
    def test_invalid_temperature_is_flagged(self, bad_t: float):
        """Non-positive or non-finite temperatures are never numerically valid or in-domain."""
        result = henry_hcp(DEFAULT_REGISTRY.get_henry("NH3"), jnp.float32(bad_t))
        assert not bool(result.numerically_valid) and not bool(result.in_domain)

    def test_batched_and_jitted_evaluation_agree(self):
        """Batched, vmapped and jitted evaluations give the same values and flags."""
        record = DEFAULT_REGISTRY.get_henry("CO2")
        t = jnp.array([270.0, 290.0, 300.0, 320.0])
        batched = henry_hcp(record, t)

        def evaluate(x: jax.Array):
            return henry_hcp(record, x)

        mapped = jax.vmap(evaluate)(t)
        compiled = jax.jit(evaluate)(t)
        assert batched.in_domain.tolist() == [False, True, True, False]
        for other in (mapped, compiled):
            assert jnp.allclose(batched.value, other.value)
            assert batched.in_domain.tolist() == other.in_domain.tolist()


class TestUnits:
    def test_co2_at_atmospheric_partial_pressure_gives_millimolar_solubility(self):
        """0.0345 mol/(L atm) times 1 atm is about 34.5 mol/m3, via 1 mol/(L atm) = 1000/101325 mol/(m3 Pa)."""
        hcp = henry_hcp(DEFAULT_REGISTRY.get_henry("CO2"), jnp.float32(298.15)).value
        concentration = equilibrium_concentration(hcp, jnp.float32(101325.0))
        assert float(concentration) == pytest.approx(0.0345 * 1000.0, rel=0.02)


class TestForwardAndInverse:
    def test_forward_and_inverse_round_trip(self):
        """Partial pressure to concentration and back recovers the partial pressure."""
        hcp = henry_hcp(DEFAULT_REGISTRY.get_henry("H2S"), jnp.linspace(280.0, 305.0, 4)).value
        pressure = jnp.array([10.0, 1e3, 5e4, 1e5])
        back = equilibrium_partial_pressure(hcp, equilibrium_concentration(hcp, pressure))
        assert jnp.allclose(back, pressure, rtol=1e-5)

    def test_zero_in_gives_exactly_zero_out(self):
        """No gas phase means no dissolved gas, and no dissolved gas means zero pressure."""
        hcp = henry_hcp(DEFAULT_REGISTRY.get_henry("NH3"), jnp.float32(293.15)).value
        assert float(equilibrium_concentration(hcp, jnp.float32(0.0))) == 0.0
        assert float(equilibrium_partial_pressure(hcp, jnp.float32(0.0))) == 0.0

    def test_signed_driving_force_is_preserved(self):
        """Concentration scales linearly with pressure, so desorption can be expressed by the caller."""
        hcp = jnp.float32(2e-3)
        assert jnp.allclose(
            equilibrium_concentration(hcp, jnp.float32(300.0)) - equilibrium_concentration(hcp, jnp.float32(500.0)),
            equilibrium_concentration(hcp, jnp.float32(-200.0)),
        )
