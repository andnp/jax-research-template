"""Fixed-yield fast closure on a sulfide/free-chlorine carrier.

The scrubber-like chemistry is a test example of the generic closure, not a plant default.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.chemistry.aqueous import DEFAULT_REGISTRY
from process_control.chemistry.basis import FamilyTotal, Retained, make_aqueous_basis
from process_control.chemistry.fast_closure import FastReactionClosure, make_fixed_yield_closure
from process_control.chemistry.network import row_component_basis
from process_control.chemistry.reactions import Reaction, compile_reactions
from process_control.components import ComponentBasis, ComponentSchema, Quantity

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
R0 = Reaction("R0", (("HS-", -1), ("OCl-", -1), ("S8(s)", Fraction(1, 8)), ("Cl-", 1), ("OH-", 1)), "per mol S consumed")
R6 = Reaction("R6", (("HS-", -1), ("OCl-", -4), ("SO4--", 1), ("Cl-", 4), ("H+", 1)), "per mol S consumed")
REACTIONS = compile_reactions(DEFAULT_REGISTRY, (R0, R6))
TRACER = ComponentBasis("g/m3", Quantity.PROXY, "tracer")

SCHEMA = ComponentSchema(
    names=(*ROWS, "tracer"),
    particulate=tuple(name == "elemental_sulfur" for name in ROWS) + (False,),
    tss_factor=tuple(32.06 if name == "elemental_sulfur" else 0.0 for name in ROWS) + (0.0,),
    bases=tuple(row_component_basis(BASIS, row) for row in BASIS.rows) + (TRACER,),
)
INDEX = {name: i for i, name in enumerate(SCHEMA.names)}


def _closure(f: float, schema: ComponentSchema = SCHEMA, channel_map: dict[str, str] | None = None) -> FastReactionClosure:
    return make_fixed_yield_closure(
        BASIS, REACTIONS, {"R0": 1.0 - f, "R6": f}, "total_sulfide", "free_chlorine", ("chloride",), schema, channel_map
    )


def _amounts(**values: float) -> jax.Array:
    base = dict.fromkeys(SCHEMA.names, 0.0) | {"sodium": 30.0, "total_carbonate": 2.0, "tracer": 4.0}
    return jnp.array([(base | values)[name] for name in SCHEMA.names])


def _named(amounts: jax.Array) -> dict[str, float]:
    return {name: float(value) for name, value in zip(SCHEMA.names, amounts, strict=True)}


class TestChartConstruction:
    @pytest.mark.parametrize("f", [0.0, 0.3, 1.0])
    def test_invariants_annihilate_the_fast_stoichiometry_with_full_rank(self, f: float):
        """U S_fast = 0 and the chart drops exactly the one eliminated progress coordinate."""
        closure = _closure(f)
        assert np.allclose(closure.invariants @ closure.stoichiometry, 0.0, atol=1e-12)
        assert np.linalg.matrix_rank(closure.invariants) == len(SCHEMA.names) - 1
        assert closure.demand == pytest.approx(1.0 + 3.0 * f)

    def test_untouched_channels_stay_identity_coordinates(self):
        """Carbonate, sodium, chlorate and the opaque tracer are stored as themselves, ahead of the invariants."""
        closure = _closure(0.5)
        untouched = [SCHEMA.names[i] for i in closure.untouched]
        assert untouched == ["total_carbonate", "sodium", "chlorate", "tracer"]
        amounts = _amounts(total_sulfide=1.0, free_chlorine=5.0)
        assert np.allclose(closure.coordinates(amounts)[:4], [2.0, 30.0, 0.0, 4.0])

    def test_renamed_bindings_give_the_same_chemistry(self):
        """Local basis names bind to differently named carrier channels, in any order, with exact bases."""
        renames = {name: f"ch_{name}" for name in ROWS}
        order = (5, 0, 7, 2, 1, 6, 3, 4)
        schema = ComponentSchema(
            names=tuple(renames[ROWS[i]] for i in order),
            particulate=tuple(ROWS[i] == "elemental_sulfur" for i in order),
            tss_factor=tuple(32.06 if ROWS[i] == "elemental_sulfur" else 0.0 for i in order),
            bases=tuple(SCHEMA.bases[i] for i in order),
        )
        permuted = _closure(0.3, schema, renames)
        reference = _closure(0.3)
        assert permuted.demand == pytest.approx(reference.demand)
        assert permuted.stoichiometry[permuted.reactant] == -1.0
        assert schema.names[permuted.oxidant] == "ch_free_chlorine"
        assert [schema.names[i] for i in permuted.oxidant_products] == ["ch_chloride"]

    def test_basis_mismatch_between_row_and_channel_is_rejected(self):
        """No unit conversion is inferred when a carrier channel has the wrong basis."""
        wrong = ComponentSchema(
            names=SCHEMA.names,
            particulate=SCHEMA.particulate,
            tss_factor=SCHEMA.tss_factor,
            bases=(TRACER, *SCHEMA.bases[1:]),
        )
        with pytest.raises(ValueError, match="basis"):
            _closure(0.5, wrong)

    @pytest.mark.parametrize("yields", [{"R0": 0.5, "R6": 0.6}, {"R0": 1.5, "R6": -0.5}, {"R0": 1.0}])
    def test_invalid_yields_are_rejected(self, yields: dict[str, float]):
        """Yields must name every reaction, lie in [0, 1] and sum to one."""
        with pytest.raises(ValueError, match="yields"):
            make_fixed_yield_closure(BASIS, REACTIONS, yields, "total_sulfide", "free_chlorine", ("chloride",), SCHEMA)


class TestFixedYieldStoichiometry:
    @pytest.mark.parametrize("f", [0.0, 0.4, 1.0])
    def test_one_mol_sulfide_consumes_oxidant_and_yields_products_by_the_fixed_yield(self, f: float):
        """1 mol S(-II) with ample FAC consumes 1+3f FAC, makes (1-f) S0, f sulfate and 1+3f chloride."""
        closure = _closure(f)
        d = 1.0 + 3.0 * f
        relaxed = _named(closure.initialize(_amounts(total_sulfide=1.0, free_chlorine=10.0)).amounts)
        assert relaxed["total_sulfide"] == pytest.approx(0.0, abs=1e-12)
        assert relaxed["free_chlorine"] == pytest.approx(10.0 - d)
        assert relaxed["elemental_sulfur"] == pytest.approx(1.0 - f)
        assert relaxed["total_sulfate"] == pytest.approx(f)
        assert relaxed["chloride"] == pytest.approx(d)

    def test_sulfur_chlorine_and_sodium_are_preserved(self):
        """Element totals are unchanged by relaxation, counting FAC and chloride together as chlorine."""
        start = _amounts(total_sulfide=2.0, free_chlorine=3.0, total_sulfate=0.5, elemental_sulfur=0.2, chloride=1.0, chlorate=0.1)
        relaxed = _named(_closure(0.4).initialize(start).amounts)
        before = _named(start)
        for group in (("total_sulfide", "total_sulfate", "elemental_sulfur"), ("free_chlorine", "chloride", "chlorate")):
            assert sum(relaxed[n] for n in group) == pytest.approx(sum(before[n] for n in group))
        for name in ("sodium", "total_carbonate", "chlorate", "tracer"):
            assert relaxed[name] == before[name]


class TestChartBranches:
    def test_oxidant_excess_branch_has_no_reactant(self):
        """I > 0: all sulfide is gone and the surplus FAC remains."""
        closure = _closure(0.5)
        coordinates = closure.initialize(_amounts(total_sulfide=1.0, free_chlorine=4.0)).coordinates
        assert float(coordinates[len(closure.untouched)]) == pytest.approx(4.0 - 2.5 * 1.0)
        amounts = _named(closure.amounts(coordinates))
        assert amounts["total_sulfide"] == 0.0
        assert amounts["free_chlorine"] == pytest.approx(1.5)

    def test_reactant_excess_branch_has_no_oxidant(self):
        """I < 0: all FAC is consumed and the remaining sulfide is the excess."""
        closure = _closure(0.5)
        amounts = _named(closure.initialize(_amounts(total_sulfide=3.0, free_chlorine=2.5)).amounts)
        assert amounts["free_chlorine"] == pytest.approx(0.0, abs=1e-12)
        assert amounts["total_sulfide"] == pytest.approx(2.0)

    def test_equality_leaves_neither_reagent(self):
        """I = 0: stoichiometric supply consumes both completely."""
        closure = _closure(1.0)
        relaxed = closure.initialize(_amounts(total_sulfide=1.0, free_chlorine=4.0))
        amounts = _named(relaxed.amounts)
        assert float(relaxed.coordinates[len(closure.untouched)]) == pytest.approx(0.0, abs=1e-12)
        assert amounts["total_sulfide"] == pytest.approx(0.0, abs=1e-12)
        assert amounts["free_chlorine"] == pytest.approx(0.0, abs=1e-12)
        assert amounts["total_sulfate"] == pytest.approx(1.0)

    def test_reconstruction_is_continuous_across_equality(self):
        """Inventories change continuously as the signed invariant crosses zero."""
        closure = _closure(0.5)
        coordinates = closure.initialize(_amounts(total_sulfide=1.0, free_chlorine=2.5)).coordinates
        at = closure.amounts(coordinates)
        for shift in (1e-9, -1e-9):
            nudged = coordinates.at[len(closure.untouched)].add(shift)
            assert np.allclose(closure.amounts(nudged), at, atol=1e-8)

    @pytest.mark.parametrize("sulfide, chlorine", [(1.0, 10.0), (3.0, 2.5), (1.0, 2.5), (0.0, 3.0), (2.0, 0.0)])
    def test_initialization_agrees_with_reconstruction_and_is_idempotent(self, sulfide: float, chlorine: float):
        """Relaxing amounts and reconstructing their invariants give one physical state; repeating changes nothing."""
        closure = _closure(0.5)
        first = closure.initialize(_amounts(total_sulfide=sulfide, free_chlorine=chlorine))
        assert np.allclose(closure.amounts(first.coordinates), first.amounts, atol=1e-12)
        again = closure.initialize(first.amounts)
        assert np.allclose(again.amounts, first.amounts, atol=1e-12)
        assert float(again.extent) == pytest.approx(0.0, abs=1e-12)

    def test_initial_extent_is_the_limiting_reactant_amount(self):
        """The one-time extent is min(n_S, n_FAC / d) in mol of sulfide."""
        closure = _closure(0.0)
        assert float(closure.initialize(_amounts(total_sulfide=1.0, free_chlorine=10.0)).extent) == pytest.approx(1.0)
        assert float(closure.initialize(_amounts(total_sulfide=3.0, free_chlorine=2.0)).extent) == pytest.approx(2.0)


class TestInvalidStates:
    def test_negative_reconstructed_product_is_rejected_not_clipped(self):
        """A chart state needing negative sulfate is infeasible; reconstruction raises instead of repairing it."""
        closure = _closure(1.0)
        coordinates = closure.initialize(_amounts(total_sulfide=1.0, free_chlorine=0.0)).coordinates
        bad = coordinates.at[len(closure.untouched) + 1 :].set(-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            closure.amounts(bad)

    def test_traced_invalid_chart_state_returns_nan(self):
        """Under jit a rejected reconstruction poisons the result with NaN."""
        closure = _closure(0.5)
        coordinates = jnp.full(len(closure.invariants), -1.0)
        assert np.isnan(jax.jit(closure.amounts)(coordinates)).all()

    def test_negative_initial_inventory_is_rejected(self):
        """Initialization needs physical, non-negative inventories."""
        with pytest.raises(ValueError, match="non-negative"):
            _closure(0.5).initialize(_amounts(total_sulfide=-1.0))

    def test_signed_invariant_is_not_clipped(self):
        """A negative I is a valid state meaning sulfide excess, and survives reconstruction exactly."""
        closure = _closure(0.0)
        coordinates = closure.initialize(_amounts(total_sulfide=3.0, free_chlorine=1.0)).coordinates
        invariant = float(coordinates[len(closure.untouched)])
        assert invariant == pytest.approx(-2.0)
        assert float(closure.amounts(coordinates)[INDEX["total_sulfide"]]) == pytest.approx(2.0)


class TestDirectionalRate:
    def _rate(self, closure: FastReactionClosure, coordinates: jax.Array, **supply: float) -> float:
        flow = jnp.array([supply.get(name, 0.0) for name in SCHEMA.names])
        return float(closure.extent_rate(coordinates, flow))

    def test_oxidant_excess_rate_is_the_reactant_supply(self):
        """With I > 0 the extent rate equals sulfide supply, in mol/h."""
        closure = _closure(0.0)
        coordinates = closure.initialize(_amounts(free_chlorine=5.0)).coordinates
        assert self._rate(closure, coordinates, total_sulfide=2.0, free_chlorine=-1.0) == pytest.approx(2.0)

    def test_reactant_excess_rate_is_the_oxidant_supply_over_demand(self):
        """With I < 0 the extent rate equals FAC supply divided by d."""
        closure = _closure(1.0)
        coordinates = closure.initialize(_amounts(total_sulfide=5.0)).coordinates
        assert self._rate(closure, coordinates, free_chlorine=8.0, total_sulfide=-3.0) == pytest.approx(2.0)

    def test_equality_rate_is_the_smaller_supply(self):
        """With both reagents absent the complementarity rate is min(supply_S, supply_FAC / d)."""
        closure = _closure(1.0)
        coordinates = closure.initialize(_amounts()).coordinates
        assert self._rate(closure, coordinates, total_sulfide=1.0, free_chlorine=8.0) == pytest.approx(1.0)
        assert self._rate(closure, coordinates, total_sulfide=3.0, free_chlorine=8.0) == pytest.approx(2.0)

    def test_negative_supply_of_an_absent_reagent_is_rejected(self):
        """An irreversible fast reaction cannot run backwards, so withdrawing an absent reagent is invalid."""
        closure = _closure(0.0)
        coordinates = closure.initialize(_amounts(free_chlorine=5.0)).coordinates
        with pytest.raises(ValueError, match="negative supply"):
            self._rate(closure, coordinates, total_sulfide=-1.0)


class TestAmountFlowTransform:
    def test_flow_transform_reacts_amount_flows_and_reports_extent_flow(self):
        """Mixed inlet amount flows relax like inventories, with the extent in mol/h."""
        closure = _closure(0.5)
        relaxed, extent_flow = closure.transform_amounts(_amounts(total_sulfide=2.0, free_chlorine=10.0))
        assert float(extent_flow) == pytest.approx(2.0)
        assert _named(relaxed)["free_chlorine"] == pytest.approx(10.0 - 2.5 * 2.0)

    def test_transform_matches_initialization_and_leaves_untouched_channels(self):
        """Inventories and flows share one transformation; opaque channels are unchanged."""
        closure = _closure(0.2)
        amounts = _amounts(total_sulfide=1.5, free_chlorine=2.0)
        initialized = closure.initialize(amounts)
        relaxed, extent = closure.transform_amounts(amounts)
        assert np.allclose(relaxed, initialized.amounts)
        assert float(extent) == pytest.approx(float(initialized.extent))
        assert float(relaxed[INDEX["tracer"]]) == 4.0
