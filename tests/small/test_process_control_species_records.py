"""Default-registry records for water, elemental sulfur and chlorate."""

import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY

# IUPAC conventional standard atomic weights.
_WEIGHTS = {"H": 1.008, "O": 15.999, "S": 32.06, "Cl": 35.45}


def _formula_mass(name: str):
    species = DEFAULT_REGISTRY.get_species(name)
    return sum(_WEIGHTS[element] * count for element, count in species.elements)


class TestAddedSpecies:
    @pytest.mark.parametrize(
        ("name", "charge", "elements"),
        [
            ("H2O", 0, {"H": 2, "O": 1}),
            ("ClO3-", -1, {"Cl": 1, "O": 3}),
            ("S8(s)", 0, {"S": 8}),
        ],
    )
    def test_formula_and_charge(self, name: str, charge: int, elements: dict[str, int]):
        """Each record carries its molecular formula and net charge."""
        species = DEFAULT_REGISTRY.get_species(name)
        assert species.charge == charge
        assert dict(species.elements) == elements

    @pytest.mark.parametrize("name", ["H2O", "ClO3-", "S8(s)"])
    def test_molar_mass_agrees_with_the_formula(self, name: str):
        """Molar mass follows from the element counts, so formula and mass cannot drift apart."""
        mass = DEFAULT_REGISTRY.get_species(name).molar_mass_g_per_mol
        assert mass == pytest.approx(_formula_mass(name), abs=0.02)

    def test_s8_is_eight_sulfur_atoms_not_one(self):
        """The 256 g/mol mass sits on an eight-atom formula; per mol S it is one eighth."""
        s8 = DEFAULT_REGISTRY.get_species("S8(s)")
        assert s8.count("S") == 8
        assert s8.molar_mass_g_per_mol / s8.count("S") == pytest.approx(32.06, abs=0.01)

    @pytest.mark.parametrize("name", ["H2O", "ClO3-", "S8(s)"])
    def test_records_cite_a_source(self, name: str):
        """Every added record names where its constants come from."""
        assert DEFAULT_REGISTRY.get_species(name).sources

    def test_chlorate_is_a_spectator_outside_every_acid_base_family(self):
        """Chlorate is a strong anion, so no family may absorb it as a form."""
        assert all("ClO3-" not in family.forms for family in DEFAULT_REGISTRY.families)
