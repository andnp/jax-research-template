"""Constructor-time validation of the chemistry registry."""

import pytest
from process_control.chemistry.species import (
    AcidBaseFamily,
    AnalyticalPool,
    HenrySolubility,
    Registry,
    Species,
)
from process_control.chemistry.thermo import AnalyticLogK, EquilibriumConstant


def _k():
    return EquilibriumConstant(AnalyticLogK(-5.0), "test", ("src",), 273.15, 323.15)


def _species(name: str, charge: int, elements: tuple[tuple[str, int], ...], mass: float = 10.0):
    return Species(name, charge, mass, elements)


HA = _species("HA", 0, (("H", 1), ("X", 1)))
A_MINUS = _species("A-", -1, (("X", 1),))
H2B = _species("H2B", 0, (("H", 2), ("Y", 1)))
HB = _species("HB-", -1, (("H", 1), ("Y", 1)))
B2 = _species("B2-", -2, (("Y", 1),))
GAS = _species("GAS", 0, (("Z", 1),))
POOL = AnalyticalPool("C_star", 0, "C")
BICARB = _species("HCO3-", -1, (("H", 1), ("C", 1), ("O", 3)))
CARB = _species("CO3--", -2, (("C", 1), ("O", 3)))


def _registry(
    species: tuple[Species, ...] = (HA, A_MINUS, H2B, HB, B2, GAS),
    pools: tuple[AnalyticalPool, ...] = (),
    families: tuple[AcidBaseFamily, ...] = (AcidBaseFamily("mono", ("HA", "A-"), (_k(),), ("s",)),),
    henry: tuple[HenrySolubility, ...] = (HenrySolubility("GAS", 1e-3, 2000.0, 278.15, 308.15, ("s",)),),
):
    return Registry(species, pools, families, henry, _k())


class TestAcceptedData:
    def test_valid_registry_exposes_typed_lookups(self):
        """Each namespace resolves its own records by name."""
        registry = _registry()
        assert registry.get_species("HA") is HA
        assert registry.get_family("mono").forms == ("HA", "A-")
        assert registry.get_henry("GAS").species == "GAS"

    def test_same_name_may_exist_in_family_and_species_namespaces(self):
        """Namespaces are separate, so a family can share a name with a species."""
        family = AcidBaseFamily("GAS", ("HA", "A-"), (_k(),), ("s",))
        assert _registry(families=(family,)).get_family("GAS") is family

    def test_diprotic_family_is_accepted(self):
        """A two-step ladder needs two constants and one H lost per step."""
        family = AcidBaseFamily("di", ("H2B", "HB-", "B2-"), (_k(), _k()), ("s",))
        _registry(families=(family,))

    def test_carbonate_pool_family_is_accepted_without_a_fake_molecule(self):
        """A pool form conserving carbon is valid beside real bicarbonate and carbonate species."""
        family = AcidBaseFamily("carbonate", ("C_star", "HCO3-", "CO3--"), (_k(), _k()), ("s",))
        registry = _registry(
            species=(BICARB, CARB, GAS), pools=(POOL,), families=(family,), henry=()
        )
        assert registry.get_form("C_star") is POOL

    def test_registry_is_hashable(self):
        """Static registries can be closed over or used as static JIT arguments."""
        assert isinstance(hash(_registry()), int)


class TestRejectedData:
    def test_duplicate_species_names(self):
        """Names must be unique across species and pools."""
        with pytest.raises(ValueError, match="duplicate"):
            _registry(species=(HA, HA, A_MINUS, GAS))

    def test_pool_name_colliding_with_a_species(self):
        """Pools share the form namespace with species."""
        with pytest.raises(ValueError, match="duplicate"):
            _registry(pools=(AnalyticalPool("HA", 0, "X"),))

    def test_duplicate_family_names(self):
        """Family names must be unique."""
        family = AcidBaseFamily("mono", ("HA", "A-"), (_k(),), ("s",))
        with pytest.raises(ValueError, match="duplicate"):
            _registry(families=(family, family))

    def test_unknown_family_form(self):
        """Families may only reference declared species or pools."""
        family = AcidBaseFamily("bad", ("HA", "nope"), (_k(),), ("s",))
        with pytest.raises(ValueError, match="unknown"):
            _registry(families=(family,))

    def test_wrong_number_of_constants(self):
        """N forms need exactly N-1 constants."""
        with pytest.raises(ValueError, match="constants"):
            AcidBaseFamily("bad", ("H2B", "HB-", "B2-"), (_k(),), ("s",))

    def test_charge_must_drop_by_exactly_one_per_step(self):
        """Skipping a charge state is rejected."""
        family = AcidBaseFamily("bad", ("H2B", "B2-"), (_k(),), ("s",))
        with pytest.raises(ValueError, match="charge"):
            _registry(families=(family,))

    def test_molecular_step_must_remove_exactly_one_hydrogen(self):
        """A charge-consistent step that changes the wrong atoms is rejected."""
        wrong = _species("W-", -1, (("H", 1), ("X", 2)))
        family = AcidBaseFamily("bad", ("HA", "W-"), (_k(),), ("s",))
        with pytest.raises(ValueError, match="one H"):
            _registry(species=(HA, wrong, GAS), families=(family,), henry=())

    def test_pool_family_members_must_conserve_the_pool_element(self):
        """A pool family whose real species lacks the conserved element is rejected."""
        family = AcidBaseFamily("bad", ("C_star", "A-"), (_k(),), ("s",))
        with pytest.raises(ValueError, match="carry"):
            _registry(pools=(AnalyticalPool("C_star", 0, "C"),), families=(family,), henry=())

    def test_family_with_duplicate_forms(self):
        """A form cannot appear twice in one family."""
        with pytest.raises(ValueError, match="duplicate"):
            AcidBaseFamily("bad", ("HA", "HA"), (_k(),), ("s",))

    def test_henry_record_for_unknown_species(self):
        """Henry records must reference a declared species."""
        record = HenrySolubility("NOPE", 1e-3, 2000.0, 278.15, 308.15, ("s",))
        with pytest.raises(ValueError, match="unknown"):
            _registry(henry=(record,))

    def test_henry_record_for_charged_species(self):
        """Henry's law applies only to neutral molecules."""
        record = HenrySolubility("A-", 1e-3, 2000.0, 278.15, 308.15, ("s",))
        with pytest.raises(ValueError, match="neutral"):
            _registry(henry=(record,))

    def test_duplicate_henry_records(self):
        """Each species has at most one Henry record."""
        record = HenrySolubility("GAS", 1e-3, 2000.0, 278.15, 308.15, ("s",))
        with pytest.raises(ValueError, match="duplicate"):
            _registry(henry=(record, record))

    @pytest.mark.parametrize("mass", [0.0, -1.0, float("nan"), float("inf")])
    def test_species_mass_must_be_finite_and_positive(self, mass: float):
        """Unphysical molar masses are rejected."""
        with pytest.raises(ValueError, match="molar mass"):
            Species("X", 0, mass, (("H", 1),))

    def test_species_charge_must_not_be_a_boolean(self):
        """A bool is not a charge even though it is an int subclass."""
        with pytest.raises(ValueError, match="integer"):
            Species("X", True, 1.0, (("H", 1),))

    def test_species_elements_must_be_unique_positive_integers(self):
        """Repeated elements and non-positive counts are rejected."""
        with pytest.raises(ValueError, match="duplicate"):
            Species("X", 0, 1.0, (("H", 1), ("H", 1)))
        with pytest.raises(ValueError, match="positive integer"):
            Species("X", 0, 1.0, (("H", 0),))

    @pytest.mark.parametrize(
        ("hcp", "b", "t_min", "sources"),
        [
            (0.0, 2000.0, 278.15, ("s",)),
            (float("nan"), 2000.0, 278.15, ("s",)),
            (1e-3, float("inf"), 278.15, ("s",)),
            (1e-3, 2000.0, 310.0, ("s",)),
            (1e-3, 2000.0, 278.15, ()),
        ],
    )
    def test_malformed_henry_records(self, hcp: float, b: float, t_min: float, sources: tuple[str, ...]):
        """Non-positive Hcp, non-finite B, bad guard ranges or missing sources are rejected."""
        with pytest.raises(ValueError):
            HenrySolubility("GAS", hcp, b, t_min, 308.15, sources)
