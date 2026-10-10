"""Aqueous basis projection: compiled S and B matrices, strong-ion reducer and acid-base invariance."""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY
from process_control.chemistry.basis import (
    FamilyTotal,
    Inert,
    Retained,
    compile_basis_stoichiometry,
    compile_transfer,
    make_aqueous_basis,
)
from process_control.chemistry.reactions import Coefficient, Reaction, compile_reactions
from process_control.chemistry.speciation import evaluate_constants, solve_ideal_charge_balance

EIGHTH = Fraction(1, 8)
ROWS = (
    "total_sulfide",
    "free_chlorine",
    "total_carbonate",
    "total_sulfate",
    "elemental_sulfur",
    "sodium",
    "chloride",
    "chlorate",
    "dissolved_oxygen",
)


def _basis(*extra_rows: Inert):
    return make_aqueous_basis(
        DEFAULT_REGISTRY,
        (
            FamilyTotal("total_sulfide", "sulfide", "S"),
            FamilyTotal("free_chlorine", "hypochlorous", "Cl"),
            FamilyTotal("total_carbonate", "carbonate", "C", extra_members=("CO2",)),
            FamilyTotal("total_sulfate", "bisulfate", "S"),
            Retained("elemental_sulfur", "S8(s)", per_mol=8.0),
            Retained("sodium", "Na+"),
            Retained("chloride", "Cl-"),
            Retained("chlorate", "ClO3-"),
            Retained("dissolved_oxygen", "O2"),
            *extra_rows,
        ),
        eliminated=("H+", "OH-", "H2O"),
    )


def _reaction(name: str, *terms: tuple[str, Coefficient]):
    return Reaction(name, terms, "per mol S consumed")


R0 = _reaction("R0", ("HS-", -1), ("OCl-", -1), ("S8(s)", EIGHTH), ("Cl-", 1), ("OH-", 1))
R0_H2S = _reaction("R0_H2S", ("H2S", -1), ("OCl-", -1), ("S8(s)", EIGHTH), ("Cl-", 1), ("H2O", 1))
R6 = _reaction("R6", ("HS-", -1), ("OCl-", -4), ("SO4--", 1), ("Cl-", 4), ("H+", 1))
R6_H2S = _reaction("R6_H2S", ("H2S", -1), ("OCl-", -4), ("SO4--", 1), ("Cl-", 4), ("H+", 2))
RPOST = _reaction("Rpost", ("S8(s)", -EIGHTH), ("OCl-", -3), ("H2O", -1), ("SO4--", 1), ("Cl-", 3), ("H+", 2))
RCHLORATE = _reaction("Rchlorate", ("OCl-", -3), ("ClO3-", 1), ("Cl-", 2))
ROXYGEN = _reaction("Roxygen", ("OCl-", -2), ("O2", 1), ("Cl-", 2))
PATHWAYS = (R0, R6, RPOST, RCHLORATE, ROXYGEN)


def _stoichiometry(*reactions: Reaction):
    return compile_basis_stoichiometry(_basis(), compile_reactions(DEFAULT_REGISTRY, reactions))


# Columns R0, R6, Rpost, Rchlorate, Roxygen over ROWS.
EXPECTED_S = np.array(
    [
        [-1, -1, 0, 0, 0],
        [-1, -4, -3, -3, -2],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 4, 3, 2, 2],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ],
    dtype=float,
)


class TestCompiledStoichiometry:
    def test_scrubber_pathways_project_to_the_documented_table(self):
        """S = P N reproduces the design's hand-derived basis table, including the 1/8 S8 times 8 S per S8."""
        assert np.allclose(_stoichiometry(*PATHWAYS), EXPECTED_S)

    def test_sulfur_and_chlorine_inventories_are_conserved(self):
        """Sulfide, sulfur and sulfate rows sum to zero, as do free chlorine, chloride and chlorate."""
        s = _stoichiometry(*PATHWAYS)
        sulfur_rows = [ROWS.index(r) for r in ("total_sulfide", "total_sulfate", "elemental_sulfur")]
        chlorine_rows = [ROWS.index(r) for r in ("free_chlorine", "chloride", "chlorate")]
        assert np.allclose(s[sulfur_rows].sum(axis=0), 0.0)
        assert np.allclose(s[chlorine_rows].sum(axis=0), 0.0)

    def test_hydrogen_sulfide_and_bisulfide_forms_compile_identically(self):
        """Forms related by an eliminated acid-base step give the same columns."""
        assert np.allclose(_stoichiometry(R0), _stoichiometry(R0_H2S))
        assert np.allclose(_stoichiometry(R6), _stoichiometry(R6_H2S))

    @pytest.mark.parametrize("yield_to_sulfate", [0.0, 0.25, 0.5, 1.0])
    def test_fixed_yield_demands_one_plus_three_f_free_chlorine(self, yield_to_sulfate: float):
        """A mix of the sulfur and sulfate endpoints consumes exactly 1 + 3f free chlorine per sulfide."""
        s = _stoichiometry(R0, R6)
        fast = (1.0 - yield_to_sulfate) * s[:, 0] + yield_to_sulfate * s[:, 1]
        assert fast[ROWS.index("free_chlorine")] == pytest.approx(-(1.0 + 3.0 * yield_to_sulfate))
        assert fast[ROWS.index("total_sulfide")] == pytest.approx(-1.0)

    def test_acid_base_redistribution_leaves_every_total_unchanged(self):
        """CO2 + OH- -> HCO3- only moves carbon between forms of one total."""
        absorb = _reaction("redistribute", ("CO2", -1), ("OH-", -1), ("HCO3-", 1))
        assert np.allclose(_stoichiometry(absorb), 0.0)


class TestTransfer:
    def test_transferred_molecules_land_on_their_rows(self):
        """B = P T converts molecule amounts to coordinates: S8 counts eight sulfur atoms, H2S one."""
        b = compile_transfer(_basis(), ("H2S", "S8(s)", "O2", "CO2"))
        assert b.shape == (len(ROWS), 4)
        assert b[ROWS.index("total_sulfide"), 0] == 1.0
        assert b[ROWS.index("elemental_sulfur"), 1] == 8.0
        assert b[ROWS.index("dissolved_oxygen"), 2] == 1.0
        assert b[ROWS.index("total_carbonate"), 3] == 1.0
        assert np.count_nonzero(b) == 4


class TestRejections:
    def test_unmapped_product_is_rejected(self):
        """A product with neither a row nor an eliminated role cannot silently disappear."""
        basis = make_aqueous_basis(
            DEFAULT_REGISTRY,
            (FamilyTotal("total_sulfide", "sulfide", "S"), FamilyTotal("free_chlorine", "hypochlorous", "Cl")),
            eliminated=("H+", "OH-", "H2O"),
        )
        reactions = compile_reactions(DEFAULT_REGISTRY, (R0,))
        with pytest.raises(ValueError, match="S8"):
            compile_basis_stoichiometry(basis, reactions)

    def test_sulfate_as_family_and_fixed_ion_is_rejected(self):
        """The bisulfate family already counts SO4--, so a retained SO4-- row would count it twice."""
        with pytest.raises(ValueError, match="more than one role"):
            make_aqueous_basis(
                DEFAULT_REGISTRY,
                (FamilyTotal("total_sulfate", "bisulfate", "S"), Retained("sulfate_ion", "SO4--")),
            )

    def test_eliminated_species_cannot_also_be_retained(self):
        """A species is either a transported row or recovered by equilibrium, not both."""
        with pytest.raises(ValueError, match="more than one role"):
            make_aqueous_basis(DEFAULT_REGISTRY, (Retained("chloride", "Cl-"),), eliminated=("Cl-",))

    def test_duplicate_row_names_are_rejected(self):
        """Row names identify transported coordinates and must be unique."""
        with pytest.raises(ValueError, match="unique"):
            make_aqueous_basis(DEFAULT_REGISTRY, (Retained("x", "Cl-"), Retained("x", "Na+")))

    def test_family_element_must_be_carried_by_every_form(self):
        """Counting carbon in the sulfide family is meaningless and rejected."""
        with pytest.raises(ValueError, match="nonzero"):
            make_aqueous_basis(DEFAULT_REGISTRY, (FamilyTotal("bad", "sulfide", "C"),))


class TestReducer:
    def test_strong_charge_is_sodium_minus_chloride_minus_chlorate(self):
        """Cations are positive and monovalent anions negative, with inert and neutral rows ignored."""
        basis = _basis(Inert("tracer"))
        c = np.zeros(len(basis.names))
        c[basis.index("sodium")] = 10.0
        c[basis.index("chloride")] = 4.0
        c[basis.index("chlorate")] = 1.0
        c[basis.index("dissolved_oxygen")] = 7.0
        c[basis.index("tracer")] = 99.0
        assert float(basis.strong_charge(jnp.asarray(c))) == pytest.approx(5.0)

    def test_family_totals_follow_the_speciation_system_order(self):
        """The totals vector is the family rows in the order of the compiled speciation system."""
        basis = _basis()
        c = jnp.arange(1.0, len(ROWS) + 1.0)
        assert [f.name for f in basis.speciation.families] == ["sulfide", "hypochlorous", "carbonate", "bisulfate"]
        assert np.allclose(basis.family_totals(c), [1.0, 2.0, 3.0, 4.0])

    def test_batched_concentrations_reduce_along_the_last_axis(self):
        """Leading batch axes pass through both reductions."""
        basis = _basis()
        c = jnp.ones((3, 2, len(ROWS)))
        assert basis.family_totals(c).shape == (3, 2, 4)
        assert basis.strong_charge(c).shape == (3, 2)

    def test_adding_sodium_chloride_leaves_ideal_ph_unchanged(self):
        """NaCl adds equal Na and Cl, so net strong charge and the charge-balance pH are unchanged."""
        basis = _basis()
        constants = evaluate_constants(basis.speciation, jnp.float32(298.15))

        def ph(sodium: float, chloride: float):
            c = np.zeros(len(ROWS), dtype=np.float32)
            c[basis.index("total_carbonate")] = 2.0
            c[basis.index("sodium")] = sodium
            c[basis.index("chloride")] = chloride
            c = jnp.asarray(c)
            return solve_ideal_charge_balance(
                basis.speciation, basis.family_totals(c), basis.strong_charge(c), constants
            ).ph

        assert jnp.allclose(ph(5.0, 1.0), ph(105.0, 101.0), atol=1e-4)
        assert float(ph(6.0, 1.0)) > float(ph(5.0, 1.0))
