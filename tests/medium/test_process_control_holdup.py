"""Variable-volume holdup: dependency contract, hydraulics, balances and fast-chart invariants.

Chemistry here is illustrative; the unit itself is generic.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.binding import NamedAccess, Requirement
from process_control.chemistry.fast_closure import FastReactionClosure
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.fields import Phase, PortSpec, PrescribedContext
from process_control.flowsheet import Mixer, Node, Port, build_flowsheet
from process_control.inventory import HoldupState
from process_control.kinetics import KineticSource
from process_control.streams import Stream
from process_control.units.contract import step_unit, unit_outlets
from process_control.units.holdup import HoldupParams, VariableHoldup
from process_control.units.hydraulics import PrismaticGeometry, RectangularWeir, weir_overflow_m3_per_h

PROXY = ComponentBasis("mol/m3", Quantity.PROXY, "x")
CONTEXT = PrescribedContext(pressure_pa=101325.0)
TEMPERATURE = jnp.asarray(293.15)
WEIR = RectangularWeir(crest_m=1.0, width_m=0.5, discharge_coefficient=0.6)
GEOMETRY = PrismaticGeometry(area_m2=4.0)
PARAMS = HoldupParams(v_min_m3=0.5, geometry=GEOMETRY, weir=WEIR)
PLAIN_PARAMS = HoldupParams(v_min_m3=0.5)


def _schema(*names: str) -> ComponentSchema:
    n = len(names)
    return ComponentSchema(names, (False,) * n, (0.0,) * n, (PROXY,) * n)


SCHEMA = _schema("a", "b")


def _stream(flow: float, *components: float, schema: ComponentSchema = SCHEMA) -> Stream:
    return Stream(jnp.asarray(flow), jnp.array(components), TEMPERATURE, jnp.asarray(101325.0), schema, Phase.LIQUID)


def _inputs(**signals: float) -> dict[str, jax.Array]:
    return {"temperature_k": TEMPERATURE} | {name: jnp.asarray(value) for name, value in signals.items()}


@dataclass(frozen=True)
class FirstOrderLoss:
    """Removes channel `a` at rate k C_a per hour."""

    rate: float

    @property
    def access(self) -> NamedAccess:
        return NamedAccess(reads=(Requirement("a", PROXY),), writes=(Requirement("a", PROXY),))

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: None) -> jax.Array:
        return -self.rate * reads


def _holdup(
    inlets: tuple[str, ...] = ("feed",),
    draws: tuple[str, ...] = ("pump",),
    overflow: str | None = None,
    sources: tuple[KineticSource[None], ...] = (),
) -> VariableHoldup:
    return VariableHoldup(SCHEMA, inlets, draws, overflow, sources, context=CONTEXT)


class TestFieldGraph:
    def _recycle(self):
        sump = _holdup(inlets=("makeup", "return"), draws=("recirc", "blowdown"), overflow="overflow")
        passthrough = Mixer(PortSpec(SCHEMA, Phase.LIQUID), ("in",), CONTEXT)
        return build_flowsheet(
            {"sump": Node(sump, PARAMS), "column": Node(passthrough, None)},
            [(Port("sump", "recirc"), Port("column", "in")), (Port("column", "out"), Port("sump", "return"))],
            {"makeup": Port("sump", "makeup")},
            {"blowdown": Port("sump", "blowdown"), "overflow": Port("sump", "overflow")},
        )

    def test_recycle_through_a_stateless_unit_compiles_acyclically(self):
        """Draw flows and composition read no inlet, so holdup -> pass-through -> holdup has a valid group order."""
        flowsheet = self._recycle()
        sump_groups = {gid.group for gid in flowsheet.plan.order if gid.node == "sump"}
        assert sump_groups == {"draw_flows", "weir_flow", "composition", "context"}

        state = {"sump": HoldupState(jnp.asarray(5.0), jnp.array([10.0, 5.0]))}
        signals = _inputs(q_recirc=6.0, q_blowdown=0.5)
        report = flowsheet.report(state, {"makeup": _stream(1.0, 0.0, 0.0)}, signals)
        assert float(report.streams[Port("column", "out")].flow) == pytest.approx(6.0)
        assert np.allclose(report.streams[Port("column", "out")].components, [2.0, 1.0])

        derivative = flowsheet.derivatives(state, {"makeup": _stream(1.0, 0.0, 0.0)}, signals)["sump"]
        assert isinstance(derivative, HoldupState)
        overflow = float(weir_overflow_m3_per_h(jnp.asarray(5.0), GEOMETRY, WEIR))
        assert float(derivative.volume_m3) == pytest.approx(1.0 - 0.5 - overflow)


class TestHydraulics:
    def test_draw_flow_and_composition_do_not_depend_on_inlet_flow(self):
        """Realized outlet flows and compositions are identical for any feed."""
        unit = _holdup(draws=("pump", "bleed"))
        state = HoldupState(jnp.asarray(5.0), jnp.array([10.0, 5.0]))
        quiet = unit_outlets(unit, state, {"feed": _stream(0.0, 0.0, 0.0)}, _inputs(q_pump=3.0, q_bleed=1.0), PLAIN_PARAMS)
        busy = unit_outlets(unit, state, {"feed": _stream(50.0, 7.0, 9.0)}, _inputs(q_pump=3.0, q_bleed=1.0), PLAIN_PARAMS)
        for name, flow in (("pump", 3.0), ("bleed", 1.0)):
            assert float(quiet[name].flow) == float(busy[name].flow) == flow
            assert np.allclose(quiet[name].components, busy[name].components)
            assert np.allclose(quiet[name].components, [2.0, 1.0])

    def test_weir_flow_is_derived_from_stored_volume_only(self):
        """The overflow equals the weir law at the stored volume and ignores feed and draw."""
        unit = _holdup(overflow="spill")
        for volume in (3.0, 5.0, 8.0):
            state = HoldupState(jnp.asarray(volume), jnp.array([1.0, 1.0]))
            outlets = unit_outlets(unit, state, {"feed": _stream(40.0, 1.0, 1.0)}, _inputs(q_pump=2.0), PARAMS)
            expected = weir_overflow_m3_per_h(jnp.asarray(volume), GEOMETRY, WEIR)
            assert float(outlets["spill"].flow) == pytest.approx(float(expected))
        assert float(unit_outlets(unit, HoldupState(jnp.asarray(3.0), jnp.ones(2)), {"feed": _stream(1.0, 0.0, 0.0)}, _inputs(q_pump=0.0), PARAMS)["spill"].flow) == 0.0

    def test_overflow_outlet_without_a_weir_is_rejected(self):
        """A holdup that declares an overflow cannot be evaluated without level geometry and a weir."""
        unit = _holdup(overflow="spill")
        state = HoldupState(jnp.asarray(5.0), jnp.ones(2))
        with pytest.raises(ValueError, match="weir"):
            unit_outlets(unit, state, {"feed": _stream(1.0, 0.0, 0.0)}, _inputs(q_pump=0.0), PLAIN_PARAMS)

    def test_overflow_removes_volume_and_solute_at_the_tank_concentration(self):
        """Weir discharge enters both the volume and the amount balance."""
        unit = _holdup(draws=("pump",), overflow="spill")
        state = HoldupState(jnp.asarray(5.0), jnp.array([10.0, 5.0]))
        q = float(weir_overflow_m3_per_h(jnp.asarray(5.0), GEOMETRY, WEIR))
        derivative = unit.derivatives(state, {"feed": _stream(0.0, 0.0, 0.0)}, _inputs(q_pump=0.0), PARAMS)
        assert float(derivative.volume_m3) == pytest.approx(-q)
        assert np.allclose(derivative.amount_coordinates, [-q * 2.0, -q * 1.0])


class TestBalances:
    def test_draw_exceeding_inflow_drains_inventory(self):
        """Stored liquid supplies the excess draw, so volume and inventory both fall."""
        unit = _holdup()
        state = HoldupState(jnp.asarray(5.0), jnp.array([10.0, 5.0]))
        derivative = unit.derivatives(state, {"feed": _stream(1.0, 2.0, 1.0)}, _inputs(q_pump=4.0), PLAIN_PARAMS)
        assert float(derivative.volume_m3) == pytest.approx(-3.0)
        assert np.allclose(derivative.amount_coordinates, [2.0 - 4.0 * 2.0, 1.0 - 4.0 * 1.0])

    def test_slow_source_is_converted_to_an_amount_once_and_matches_the_analytic_solution(self):
        """Filling with clean water while a first-order source acts gives n = n0 exp(-k t) and C = n / V(t)."""
        unit = _holdup(sources=(KineticSource.bind(FirstOrderLoss(0.3), None, SCHEMA),))
        state = HoldupState(jnp.asarray(4.0), jnp.array([8.0, 2.0]))
        feed = {"feed": _stream(2.0, 0.0, 0.0)}
        signals = _inputs(q_pump=0.0)

        derivative = unit.derivatives(state, feed, signals, PLAIN_PARAMS)
        assert float(derivative.amount_coordinates[0]) == pytest.approx(-0.3 * 8.0)

        final = step_unit(unit, state, feed, signals, PLAIN_PARAMS, jnp.asarray(1.0), 20)
        assert float(final.volume_m3) == pytest.approx(4.0 + 2.0)
        assert float(final.amount_coordinates[0]) == pytest.approx(8.0 * np.exp(-0.3), rel=1e-6)
        assert float(final.amount_coordinates[1]) == pytest.approx(2.0)

    def test_inlet_order_does_not_change_the_balance(self):
        """Permuting the declared inlets, with the same streams, leaves the derivative unchanged within summation tolerance."""
        streams = {"x": _stream(1.5, 1.0, 0.0), "y": _stream(0.25, 0.0, 6.0), "z": _stream(3.0, 2.0, 2.0)}
        state = HoldupState(jnp.asarray(5.0), jnp.array([10.0, 5.0]))
        derivatives = [
            _holdup(inlets=order).derivatives(state, streams, _inputs(q_pump=1.0), PLAIN_PARAMS) for order in (("x", "y", "z"), ("z", "x", "y"), ("y", "z", "x"))
        ]
        for other in derivatives[1:]:
            assert np.allclose(other.amount_coordinates, derivatives[0].amount_coordinates, rtol=0, atol=1e-12)
            assert float(other.volume_m3) == pytest.approx(float(derivatives[0].volume_m3), abs=1e-12)

    def test_invalid_volume_is_rejected_before_division(self):
        """A holdup at its declared minimum volume raises instead of dividing by it."""
        unit = _holdup()
        with pytest.raises(ValueError, match="holdup volume"):
            unit.derivatives(HoldupState(jnp.asarray(0.5), jnp.ones(2)), {"feed": _stream(1.0, 0.0, 0.0)}, _inputs(q_pump=0.0), PLAIN_PARAMS)


class TestFastChart:
    NAMES = ("sulfide", "chlorine", "sulfur", "sulfate", "chloride", "sodium")
    F = 0.5

    def _unit(self):
        d = 1.0 + 3.0 * self.F
        stoichiometry = np.array([-1.0, -d, 1.0 - self.F, self.F, d, 0.0])
        schema = _schema(*self.NAMES)
        closure = FastReactionClosure(schema, stoichiometry, reactant=0, oxidant=1, oxidant_products=(4,))
        return schema, closure, VariableHoldup(schema, ("feed", "feed2"), ("pump",), chart=closure, context=CONTEXT)

    @pytest.mark.parametrize("sulfide, chlorine", [(0.0, 6.0), (4.0, 0.0)])
    def test_derivative_preserves_sulfur_and_chlorine_balances(self, sulfide: float, chlorine: float):
        """Sulfur and chlorine totals change only by transport, whichever reagent is in excess, with reconstructed outlets."""
        schema, closure, unit = self._unit()
        amounts = jnp.array([sulfide, chlorine, 1.0, 0.5, 2.0, 30.0])
        start = closure.initialize(amounts)
        state = HoldupState(jnp.asarray(5.0), start.coordinates)
        feeds = {
            "feed": _stream(1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, schema=schema),
            "feed2": _stream(0.5, 0.0, 4.0, 0.2, 0.1, 0.4, 0.0, schema=schema),
        }
        derivative = unit.derivatives(state, feeds, _inputs(q_pump=2.0), PLAIN_PARAMS)

        physical = np.asarray(start.amounts)
        concentration = physical / 5.0
        sulfur, chlorine_total = np.array([1, 0, 1, 1, 0, 0.0]), np.array([0, 1, 0, 0, 1, 0.0])

        def expected(weights: np.ndarray) -> float:
            fed = sum(float(s.flow) * float(weights @ np.asarray(s.components)) for s in feeds.values())
            return fed - 2.0 * float(weights @ concentration)

        coordinates = np.asarray(derivative.amount_coordinates)
        nu = len(closure.untouched)
        sulfur_rate = coordinates[nu + 1] + coordinates[nu + 2]
        chlorine_rate = coordinates[nu + 3]
        assert sulfur_rate == pytest.approx(expected(sulfur))
        assert chlorine_rate == pytest.approx(expected(chlorine_total))
        assert float(derivative.volume_m3) == pytest.approx(1.5 - 2.0)

    def test_outlets_carry_the_reconstructed_physical_concentration(self):
        """The composition group reconstructs relaxed inventories and divides by the same volume."""
        schema, closure, unit = self._unit()
        start = closure.initialize(jnp.array([1.0, 6.0, 0.0, 0.0, 0.0, 30.0]))
        state = HoldupState(jnp.asarray(5.0), start.coordinates)
        feeds = {name: _stream(0.0, *[0.0] * 6, schema=schema) for name in ("feed", "feed2")}
        outlet = unit_outlets(unit, state, feeds, _inputs(q_pump=1.0), PLAIN_PARAMS)["pump"]
        assert np.allclose(outlet.components, np.asarray(start.amounts) / 5.0)
        assert float(outlet.components[0]) == 0.0
