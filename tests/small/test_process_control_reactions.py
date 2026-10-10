"""Reaction records are accepted only when exactly balanced over molecular registry species."""

from fractions import Fraction
from typing import cast

import numpy as np
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY, WATER_AUTOIONIZATION
from process_control.chemistry.reactions import Coefficient, Reaction, compile_reactions, validate_reaction
from process_control.chemistry.species import Registry, Species

EIGHTH = Fraction(1, 8)


def _reaction(name: str, *terms: tuple[str, Coefficient]):
    return Reaction(name, tuple(terms), "per mol S consumed")


R0 = _reaction("R0", ("HS-", -1), ("OCl-", -1), ("S8(s)", EIGHTH), ("Cl-", 1), ("OH-", 1))
R6 = _reaction("R6", ("HS-", -1), ("OCl-", -4), ("SO4--", 1), ("Cl-", 4), ("H+", 1))
RPOST = _reaction("Rpost", ("S8(s)", -EIGHTH), ("OCl-", -3), ("H2O", -1), ("SO4--", 1), ("Cl-", 3), ("H+", 2))
RCHLORATE = _reaction("Rchlorate", ("OCl-", -3), ("ClO3-", 1), ("Cl-", 2))
ROXYGEN = _reaction("Roxygen", ("OCl-", -2), ("O2", 1), ("Cl-", 2))
BALANCED = (R0, R6, RPOST, RCHLORATE, ROXYGEN)


class TestBalancedReactions:
    @pytest.mark.parametrize("reaction", BALANCED, ids=lambda r: r.name)
    def test_scrubber_pathways_validate(self, reaction: Reaction):
        """The oxidation, post-oxidation, chlorate and oxygen pathways are balanced in every element and charge."""
        validate_reaction(DEFAULT_REGISTRY, reaction)

    def test_hydrogen_sulfide_forms_validate(self):
        """The neutral H2S forms of the sulfur and sulfate endpoints balance with water and protons."""
        validate_reaction(
            DEFAULT_REGISTRY,
            _reaction("R0_H2S", ("H2S", -1), ("OCl-", -1), ("S8(s)", EIGHTH), ("Cl-", 1), ("H2O", 1)),
        )
        validate_reaction(
            DEFAULT_REGISTRY,
            _reaction("R6_H2S", ("H2S", -1), ("OCl-", -4), ("SO4--", 1), ("Cl-", 4), ("H+", 2)),
        )

    def test_integer_coefficients_become_exact_fractions(self):
        """Coefficients are stored exactly whatever numeric type declared them."""
        assert all(isinstance(c, Fraction) for _, c in R6.terms)
        assert R0.coefficient("S8(s)") == Fraction(1, 8)
        assert R0.coefficient("O2") == 0


class TestRejectedReactions:
    def test_missing_product_fails(self):
        """Dropping the hydroxide product leaves hydrogen, oxygen and charge unbalanced."""
        wrong = _reaction("bad", ("HS-", -1), ("OCl-", -1), ("S8(s)", EIGHTH), ("Cl-", 1))
        with pytest.raises(ValueError, match="elements"):
            validate_reaction(DEFAULT_REGISTRY, wrong)

    def test_charge_imbalance_with_balanced_elements_fails(self):
        """Two species with one formula but different charge expose a pure charge error."""
        registry = Registry(
            (Species("X", 0, 10.0, (("X", 1),)), Species("X+", 1, 10.0, (("X", 1),))),
            (),
            (),
            (),
            WATER_AUTOIONIZATION,
        )
        with pytest.raises(ValueError, match="charge"):
            validate_reaction(registry, _reaction("bad", ("X", -1), ("X+", 1)))

    def test_wrong_element_balance_fails(self):
        """Omitting the water of Rpost breaks the hydrogen and oxygen balance."""
        wrong = _reaction("bad", ("S8(s)", -EIGHTH), ("OCl-", -3), ("SO4--", 1), ("Cl-", 3), ("H+", 2))
        with pytest.raises(ValueError, match="elements"):
            validate_reaction(DEFAULT_REGISTRY, wrong)

    def test_s8_without_the_factor_of_eight_fails(self):
        """Writing one S8 per mol S counts eight sulfur atoms where one is consumed."""
        wrong = _reaction("bad", ("HS-", -1), ("OCl-", -1), ("S8(s)", 1), ("Cl-", 1), ("OH-", 1))
        with pytest.raises(ValueError, match="elements"):
            validate_reaction(DEFAULT_REGISTRY, wrong)

    def test_analytical_pool_is_rejected(self):
        """CO2* has no molecular formula and cannot enter a complete balance."""
        with pytest.raises(ValueError, match="analytical pool"):
            validate_reaction(
                DEFAULT_REGISTRY,
                _reaction("bad", ("CO2_star", -1), ("OH-", -1), ("HCO3-", 1)),
            )

    def test_unknown_species_is_rejected(self):
        """A species absent from the registry cannot be balanced."""
        with pytest.raises(ValueError, match="unknown species"):
            validate_reaction(DEFAULT_REGISTRY, _reaction("bad", ("ClO2-", -1), ("Cl-", 1)))

    def test_duplicate_terms_are_rejected(self):
        """A species listed twice is ambiguous, even when the sum would balance."""
        with pytest.raises(ValueError, match="duplicate"):
            _reaction("bad", ("OCl-", -1), ("OCl-", 1))

    @pytest.mark.parametrize("coefficient", [0, Fraction(0), 0.5, float("nan"), float("inf"), True])
    def test_zero_or_inexact_coefficients_are_rejected(self, coefficient: object):
        """Zero, float, nonfinite and boolean coefficients are not exact stoichiometry."""
        with pytest.raises(ValueError, match="coefficient"):
            Reaction("bad", (("OCl-", cast(Coefficient, coefficient)),), "x")

    def test_empty_reaction_is_rejected(self):
        """A reaction with no terms has no content to balance."""
        with pytest.raises(ValueError, match="at least one term"):
            Reaction("bad", (), "x")


class TestCompiledStoichiometry:
    def test_matrix_columns_follow_reactions_over_species_in_first_use_order(self):
        """The float matrix holds each reaction's signed coefficients row by species."""
        compiled = compile_reactions(DEFAULT_REGISTRY, BALANCED)

        assert compiled.stoichiometry.shape == (len(compiled.species), len(BALANCED))
        row = compiled.species.index("S8(s)")
        assert np.allclose(compiled.stoichiometry[row], [0.125, 0.0, -0.125, 0.0, 0.0])
        row = compiled.species.index("OCl-")
        assert np.allclose(compiled.stoichiometry[row], [-1, -4, -3, -3, -2])

    def test_compiled_reactions_conserve_every_element_and_charge(self):
        """The compiled matrix is an independent balance check: formula and charge vectors annihilate it."""
        compiled = compile_reactions(DEFAULT_REGISTRY, BALANCED)
        forms = [DEFAULT_REGISTRY.get_species(name) for name in compiled.species]

        charges = np.array([f.charge for f in forms], dtype=float)
        assert np.allclose(charges @ compiled.stoichiometry, 0.0)
        for element in {e for f in forms for e, _ in f.elements}:
            counts = np.array([f.count(element) for f in forms], dtype=float)
            assert np.allclose(counts @ compiled.stoichiometry, 0.0), element

    def test_invalid_reaction_blocks_compilation(self):
        """One unbalanced record rejects the whole set."""
        wrong = _reaction("bad", ("OCl-", -1), ("Cl-", 1))
        with pytest.raises(ValueError, match="bad"):
            compile_reactions(DEFAULT_REGISTRY, (R0, wrong))

    def test_duplicate_reaction_names_are_rejected(self):
        """Reaction names key diagnostics, so they must be unique within a set."""
        with pytest.raises(ValueError, match="duplicate reaction"):
            compile_reactions(DEFAULT_REGISTRY, (R0, R0))
