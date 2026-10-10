"""Aqueous reaction network as a kinetics source, with illustrative slow first-order test laws.

The laws below are test fixtures that exercise the machinery; they are not plant chemistry.
"""

from fractions import Fraction
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY
from process_control.chemistry.basis import FamilyTotal, Retained, make_aqueous_basis
from process_control.chemistry.network import TEMPERATURE_EXTRAPOLATION, make_aqueous_reaction_network
from process_control.chemistry.reactions import Reaction, compile_reactions
from process_control.chemistry.speciation import evaluate_constants, fractions
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.kinetics import KineticSource

T25 = 298.15
ROWS = (
    "total_sulfide",
    "free_chlorine",
    "total_carbonate",
    "total_sulfate",
    "elemental_sulfur",
    "sodium",
    "chloride",
    "chlorate",
)
SULFUR = ("total_sulfide", "total_sulfate", "elemental_sulfur")
CHLORINE = ("free_chlorine", "chloride", "chlorate")

BASIS = make_aqueous_basis(
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
    ),
    eliminated=("H+", "OH-", "H2O"),
)
R0 = Reaction(
    "R0",
    (("HS-", -1), ("OCl-", -1), ("S8(s)", Fraction(1, 8)), ("Cl-", 1), ("OH-", 1)),
    "per mol S consumed",
)
RCHLORATE = Reaction("Rchlorate", (("OCl-", -3), ("ClO3-", 1), ("Cl-", 2)), "per mol chlorate formed")
REACTIONS = compile_reactions(DEFAULT_REGISTRY, (R0, RCHLORATE))


class Params(NamedTuple):
    k_sulfide: jax.Array
    k_chlorate: jax.Array


class FirstOrder:
    """r = k [species], with the rate constant read from the parameter pack."""

    def __init__(self, species: str, field: str):
        self.species = species
        self.field = field

    @property
    def reads(self):
        return (self.species,)

    def extent_rate(self, concentrations: jax.Array, temperature_k: jax.Array, params: Params):
        return params._asdict()[self.field] * concentrations[..., 0]


PARAMS = Params(jnp.float32(1e-4), jnp.float32(5e-5))
NETWORK = make_aqueous_reaction_network(
    BASIS, REACTIONS, (FirstOrder("HS-", "k_sulfide"), FirstOrder("OCl-", "k_chlorate"))
)


def _state(**overrides: float):
    values = {
        "total_sulfide": 2.0,
        "free_chlorine": 6.0,
        "total_carbonate": 3.0,
        "total_sulfate": 0.5,
        "elemental_sulfur": 0.2,
        "sodium": 24.0,
        "chloride": 8.0,
        "chlorate": 0.1,
    }
    values.update(overrides)
    return jnp.array([values[name] for name in ROWS], dtype=jnp.float32)


def _rates(state: jax.Array, temperature_k: float = T25):
    return NETWORK.rates(state, jnp.float32(temperature_k), PARAMS)


def _by_name(rates: jax.Array):
    return dict(zip((w.name for w in NETWORK.access.writes), np.asarray(rates), strict=True))


def _form_concentration(state: jax.Array, family_index: int, form: int, total: str):
    """Independent form concentration from the system's own constants and the reported pH."""
    ph = NETWORK.evaluate(state, jnp.float32(T25), PARAMS).speciation.ph
    log_ka = evaluate_constants(BASIS.speciation, jnp.float32(T25)).log_ka
    alpha = fractions(log_ka[family_index : family_index + 1], ph)
    return float(state[ROWS.index(total)] * alpha[form])


class TestSources:
    def test_sources_match_hand_computed_extents(self):
        """Rates equal 3600 S r, with r = k [form] taken from an independent fraction calculation."""
        state = _state()
        evaluation = NETWORK.evaluate(state, jnp.float32(T25), PARAMS)
        assert bool(evaluation.usable)

        r_sulfide = 1e-4 * _form_concentration(state, 0, 1, "total_sulfide")
        r_chlorate = 5e-5 * _form_concentration(state, 1, 1, "free_chlorine")
        assert np.allclose(evaluation.extent_rates, [r_sulfide, r_chlorate], rtol=1e-4)

        expected = {
            "total_sulfide": -r_sulfide,
            "free_chlorine": -r_sulfide - 3 * r_chlorate,
            "elemental_sulfur": r_sulfide,
            "chloride": r_sulfide + 2 * r_chlorate,
            "chlorate": r_chlorate,
        }
        rates = _by_name(evaluation.rates_per_hour)
        assert set(rates) == set(expected)
        for name, value in expected.items():
            assert rates[name] == pytest.approx(3600.0 * value, rel=1e-4)

    def test_sulfur_chlorine_and_sodium_totals_are_conserved(self):
        """Rates sum to zero over each conserved inventory, counting S8 as eight sulfur atoms."""
        rates = _by_name(_rates(_state()))
        assert sum(rates.get(name, 0.0) for name in SULFUR) == pytest.approx(0.0, abs=1e-6)
        assert sum(rates.get(name, 0.0) for name in CHLORINE) == pytest.approx(0.0, abs=1e-6)
        assert "sodium" not in rates

    def test_a_law_may_read_protons_and_hydroxide(self):
        """H+ and OH- have no basis row, yet a law reading them gets mol/m3 values with [H+][OH-] = Kw."""

        class IonProduct:
            reads = ("H+", "OH-")

            def extent_rate(self, concentrations: jax.Array, temperature_k: jax.Array, params: Params):
                return concentrations[..., 0] * concentrations[..., 1]

        network = make_aqueous_reaction_network(BASIS, compile_reactions(DEFAULT_REGISTRY, (R0,)), (IonProduct(),))
        evaluation = network.evaluate(_state(), jnp.float32(T25), PARAMS)
        log_kw = float(evaluate_constants(BASIS.speciation, jnp.float32(T25)).log_kw)
        assert float(evaluation.extent_rates[0]) == pytest.approx(1000.0**2 * 10.0**log_kw, rel=1e-3)

    def test_access_declares_exact_bases_for_every_active_row(self):
        """Reads cover all chemistry-active rows; writes only those reactions change, with matching bases."""
        assert [r.name for r in NETWORK.access.reads] == list(ROWS)
        assert [w.name for w in NETWORK.access.writes] == [
            "total_sulfide",
            "free_chlorine",
            "elemental_sulfur",
            "chloride",
            "chlorate",
        ]
        by_name = {r.name: r.basis for r in NETWORK.access.reads}
        assert by_name["elemental_sulfur"] == ComponentBasis("mol/m3", Quantity.ELEMENT, "S")
        assert by_name["sodium"] == ComponentBasis("mol/m3", Quantity.SPECIES, "Na+")


class TestFailureIsNotZero:
    @pytest.mark.parametrize(
        "state",
        [_state(total_sulfide=-1.0), _state(sodium=5000.0), _state(total_sulfide=float("nan"))],
        ids=["negative-total", "beyond-sulfide-ph-ceiling", "nan-input"],
    )
    def test_unsupported_speciation_gives_nan_rates(self, state: jax.Array):
        """Invalid inputs and pH outside the sulfide ladder give NaN rates and extents, never zero."""
        evaluation = NETWORK.evaluate(state, jnp.float32(T25), PARAMS)
        assert not bool(evaluation.usable)
        assert bool(jnp.all(jnp.isnan(evaluation.rates_per_hour)))
        assert bool(jnp.all(jnp.isnan(evaluation.extent_rates)))

    def test_temperature_outside_fitted_range_is_unusable_unless_extrapolation_is_named(self):
        """The strict policy rejects a constant outside its fitted range; the named policy admits it."""
        cold = jnp.float32(263.15)
        state = _state(sodium=18.0)
        assert not bool(NETWORK.evaluate(state, cold, PARAMS).usable)

        permissive = make_aqueous_reaction_network(
            BASIS,
            REACTIONS,
            (FirstOrder("HS-", "k_sulfide"), FirstOrder("OCl-", "k_chlorate")),
            TEMPERATURE_EXTRAPOLATION,
        )
        evaluation = permissive.evaluate(state, cold, PARAMS)
        assert bool(evaluation.usable)
        assert bool(jnp.all(jnp.isfinite(evaluation.rates_per_hour)))
        assert permissive.policy.name == "temperature-extrapolation"


class TestConstruction:
    def test_laws_must_match_reactions(self):
        """Each reaction needs exactly one rate law."""
        with pytest.raises(ValueError, match="one rate law per reaction"):
            make_aqueous_reaction_network(BASIS, REACTIONS, (FirstOrder("HS-", "k_sulfide"),))

    def test_law_reads_must_be_suppliable(self):
        """A law reading a species the basis cannot provide is rejected at build time."""
        with pytest.raises(ValueError, match="cannot supply"):
            make_aqueous_reaction_network(
                BASIS, compile_reactions(DEFAULT_REGISTRY, (R0,)), (FirstOrder("NH3", "k_sulfide"),)
            )

    def test_unmapped_reaction_species_is_rejected(self):
        """A reaction producing a species with no row or eliminated role cannot join the network."""
        oxygen = Reaction("Roxygen", (("OCl-", -2), ("O2", 1), ("Cl-", 2)), "per mol O2")
        with pytest.raises(ValueError, match="O2"):
            make_aqueous_reaction_network(BASIS, compile_reactions(DEFAULT_REGISTRY, (oxygen,)), (FirstOrder("OCl-", "k_sulfide"),))


def _carrier(names: tuple[str, ...]):
    return ComponentSchema(
        names,
        tuple(name == "elemental_sulfur" for name in names),
        tuple(32.06 if name == "elemental_sulfur" else 0.0 for name in names),
        tuple(_carrier_basis(name) for name in names),
    )


def _carrier_basis(name: str):
    by_name = {r.name: r.basis for r in NETWORK.access.reads}
    return by_name.get(name, ComponentBasis("g/m3", Quantity.PROXY, "tracer"))


class TestKineticSourceBinding:
    def test_extended_shuffled_carrier_gives_the_same_sources_and_leaves_extras_untouched(self):
        """Binding by name makes carrier order and unrelated channels irrelevant."""
        state = _state()
        base = KineticSource.bind(NETWORK, PARAMS, _carrier(ROWS))
        base_total = base.add_to(jnp.zeros_like(state), state, {}, jnp.float32(T25))

        shuffled = ("chloride", "tracer", "sodium", "total_sulfate", "chlorate", "total_carbonate", "free_chlorine", "elemental_sulfur", "total_sulfide")
        carrier = _carrier(shuffled)
        values = {**dict(zip(ROWS, np.asarray(state), strict=True)), "tracer": 123.0}
        extended = jnp.array([values[name] for name in shuffled], dtype=jnp.float32)
        source = KineticSource.bind(NETWORK, PARAMS, carrier)
        total = source.add_to(jnp.zeros_like(extended), extended, {}, jnp.float32(T25))

        for name in ROWS:
            assert float(total[shuffled.index(name)]) == pytest.approx(float(base_total[ROWS.index(name)]), rel=1e-5)
        assert float(total[shuffled.index("tracer")]) == 0.0


class TestJaxTransformations:
    def test_jit_matches_eager(self):
        """Compiling the source does not change its rates."""
        state = _state()

        def rates(s: jax.Array, t: jax.Array, p: Params):
            return NETWORK.rates(s, t, p)

        assert jnp.allclose(jax.jit(rates)(state, jnp.float32(T25), PARAMS), _rates(state), rtol=1e-5)

    def test_vmap_over_states_and_temperatures_matches_a_loop(self):
        """Batched evaluation agrees lane by lane, including a failed lane staying NaN."""
        states = jnp.stack([_state(), _state(total_sulfide=0.5), _state(sodium=5000.0)])
        temperatures = jnp.array([T25, 285.0, T25], dtype=jnp.float32)

        def rates(s: jax.Array, t: jax.Array):
            return NETWORK.rates(s, t, PARAMS)

        batched = jax.vmap(rates)(states, temperatures)
        looped = jnp.stack([_rates(s, float(t)) for s, t in zip(states, temperatures, strict=True)])

        assert batched.shape == (3, len(NETWORK.access.writes))
        assert jnp.allclose(batched, looped, rtol=1e-5, equal_nan=True)
        assert bool(jnp.all(jnp.isnan(batched[2])))
        assert bool(jnp.all(jnp.isfinite(batched[:2])))

    def test_rates_are_differentiable_in_parameters(self):
        """The gradient of a sulfide rate with respect to its rate constant equals rate over constant."""
        def sulfide_rate(k: jax.Array):
            return NETWORK.rates(_state(), jnp.float32(T25), Params(k, PARAMS.k_chlorate))[0]

        gradient = jax.grad(sulfide_rate)(PARAMS.k_sulfide)
        assert float(gradient) == pytest.approx(float(sulfide_rate(PARAMS.k_sulfide) / PARAMS.k_sulfide), rel=1e-4)
