"""Default aqueous chemistry data checked against independently published values."""

import jax.numpy as jnp
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY, WATER_AUTOIONIZATION
from process_control.chemistry.species import AnalyticalPool

_ATOMIC_WEIGHTS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990, "S": 32.06, "Cl": 35.45}


def _celsius(c: float):
    return jnp.float32(c + 273.15)


def _pka(family: str, step: int, celsius: float):
    constant = DEFAULT_REGISTRY.get_family(family).log_ka[step]
    return -float(constant.evaluate(_celsius(celsius)).value)


class TestPublishedAcidConstants:
    @pytest.mark.parametrize(("celsius", "published"), [(5.0, 7.754), (25.0, 7.537), (35.0, 7.463)])
    def test_hypochlorous_pka_matches_morris_table(self, celsius: float, published: float):
        """HOCl pKa agrees with Morris (1966) measured values."""
        assert _pka("hypochlorous", 0, celsius) == pytest.approx(published, abs=0.01)

    @pytest.mark.parametrize(("celsius", "published"), [(0.0, 10.0813), (25.0, 9.2449), (50.0, 8.5387)])
    def test_ammonium_pka_matches_bates_pinching_table(self, celsius: float, published: float):
        """NH4+ pKa agrees with the Bates & Pinching (1949) tabulated values."""
        assert _pka("ammonium", 0, celsius) == pytest.approx(published, abs=0.003)

    def test_carbonate_pka_at_25c_matches_reference_values(self):
        """Successive carbonate pKa values reverse the association constants correctly."""
        assert _pka("carbonate", 0, 25.0) == pytest.approx(6.352, abs=0.005)
        assert _pka("carbonate", 1, 25.0) == pytest.approx(10.329, abs=0.005)

    def test_sulfide_pka1_is_about_seven_at_25c(self):
        """Coarse check against the room-temperature pKa1 near 7.0 of Ellis & Giggenbach."""
        assert _pka("sulfide", 0, 25.0) == pytest.approx(7.0, abs=0.05)

    def test_bisulfate_is_a_strong_second_acid_at_25c(self):
        """HSO4- pKa2 of sulfuric acid is about 1.99 at 25 C."""
        assert _pka("bisulfate", 0, 25.0) == pytest.approx(1.99, abs=0.05)

    @pytest.mark.parametrize(("celsius", "published"), [(0.0, 14.95), (25.0, 13.99), (50.0, 13.26)])
    def test_water_pkw_matches_modern_values(self, celsius: float, published: float):
        """The temperature-only Kw fit stays close to modern pKw values."""
        pkw = -float(WATER_AUTOIONIZATION.evaluate(_celsius(celsius)).value)
        assert pkw == pytest.approx(published, abs=0.02)

    def test_acids_get_stronger_or_weaker_in_the_expected_direction(self):
        """Ammonium and hypochlorous acids strengthen when heated, as for endothermic dissociation."""
        for family in ("ammonium", "hypochlorous"):
            assert _pka(family, 0, 5.0) > _pka(family, 0, 35.0)


class TestDomains:
    def test_ranges_follow_the_cited_fits(self):
        """Room temperature is in-domain everywhere; HOCl ends at 35 C and sulfide at 45 C."""
        hocl = DEFAULT_REGISTRY.get_family("hypochlorous").log_ka[0]
        sulfide = DEFAULT_REGISTRY.get_family("sulfide").log_ka[0]
        assert bool(hocl.evaluate(_celsius(25.0)).in_domain)
        assert bool(hocl.evaluate(_celsius(35.0)).in_domain)
        assert not bool(hocl.evaluate(_celsius(45.0)).in_domain)
        assert bool(sulfide.evaluate(_celsius(45.0)).in_domain)
        assert not bool(sulfide.evaluate(_celsius(50.0)).in_domain)

    def test_sulfide_family_records_its_ph_ceiling(self):
        """The S--free sulfide ladder advertises a model pH limit; others do not."""
        assert DEFAULT_REGISTRY.get_family("sulfide").model_ph_max == 12.0
        assert DEFAULT_REGISTRY.get_family("carbonate").model_ph_max is None

    def test_every_constant_carries_a_source(self):
        """Each equilibrium constant cites at least one source."""
        for family in DEFAULT_REGISTRY.families:
            assert family.sources
            assert all(k.sources for k in family.log_ka)


class TestHenryRows:
    @pytest.mark.parametrize(
        ("species", "hcp", "b"),
        [("H2S", 1.0e-3, 2100.0), ("CO2", 3.4e-4, 2300.0), ("O2", 1.3e-5, 1500.0), ("NH3", 5.9e-1, 4200.0), ("CH4", 1.4e-5, 1600.0)],
    )
    def test_reference_rows_match_the_sander_compilation(self, species: str, hcp: float, b: float):
        """Hcp at 298.15 K and the temperature slope are the assessed rows as printed."""
        record = DEFAULT_REGISTRY.get_henry(species)
        assert record.t_ref_k == 298.15
        assert record.hcp_ref_mol_per_m3_pa == hcp
        assert record.b_kelvin == b

    @pytest.mark.parametrize(
        ("species", "mol_per_l_atm"), [("CO2", 0.034), ("O2", 1.3e-3), ("CH4", 1.4e-3), ("H2S", 0.10), ("NH3", 60.0)]
    )
    def test_rows_agree_with_textbook_values_in_mol_per_litre_atmosphere(self, species: str, mol_per_l_atm: float):
        """Converting with 1 mol/(L atm) = 1000/101325 mol/(m3 Pa) reproduces familiar solubilities."""
        unit = 1000.0 / 101325.0
        record = DEFAULT_REGISTRY.get_henry(species)
        assert record.hcp_ref_mol_per_m3_pa / unit == pytest.approx(mol_per_l_atm, rel=0.05)

    def test_guard_range_is_5_to_35_celsius(self):
        """The design guard spans 278.15 to 308.15 K for every record."""
        for record in DEFAULT_REGISTRY.henry:
            assert (record.guard_t_min_k, record.guard_t_max_k) == (278.15, 308.15)


class TestSpecies:
    def test_default_registry_constructs_with_expected_families(self):
        """Default data passes registry validation and exposes the five families."""
        assert {f.name for f in DEFAULT_REGISTRY.families} == {
            "carbonate",
            "sulfide",
            "hypochlorous",
            "ammonium",
            "bisulfate",
        }

    def test_molar_masses_match_formula_atomic_weights(self):
        """Each tabulated molar mass equals the sum of its element weights within rounding."""
        for species in DEFAULT_REGISTRY.species:
            expected = sum(_ATOMIC_WEIGHTS[e] * n for e, n in species.elements)
            assert species.molar_mass_g_per_mol == pytest.approx(expected, abs=0.02), species.name

    def test_carbonate_neutral_form_is_a_pool_not_a_molecule(self):
        """CO2* has no molar mass, while molecular CO2 is a separate species."""
        pool = DEFAULT_REGISTRY.get_form("CO2_star")
        assert isinstance(pool, AnalyticalPool)
        assert pool.charge == 0 and pool.conserved_element == "C"
        assert DEFAULT_REGISTRY.get_species("CO2").charge == 0
