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
from process_control.events import Continuation
from process_control.fields import LocalField, Phase, PortSpec, PrescribedContext, StreamField
from process_control.flowsheet import Mixer, Node, Port, build_flowsheet
from process_control.inventory import HoldupState
from process_control.kinetics import KineticSource
from process_control.streams import Stream
from process_control.units.contract import step_unit, unit_outlets
from process_control.units.holdup import (
    CAPACITY_ACTIVE,
    DRAW_INTERLOCK,
    HOLDUP_CHECKS,
    HoldupParams,
    OverflowLaw,
    VariableHoldup,
    holdup_policy,
)
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

    def test_empty_holdup_is_rejected_before_division(self):
        """A holdup holding no liquid raises instead of dividing by its volume."""
        unit = _holdup()
        with pytest.raises(ValueError, match="holdup volume"):
            unit.derivatives(HoldupState(jnp.asarray(0.0), jnp.ones(2)), {"feed": _stream(1.0, 0.0, 0.0)}, _inputs(q_pump=0.0), PLAIN_PARAMS)


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


CAPACITY_PARAMS = HoldupParams(v_min_m3=0.5, capacity_m3=10.0)


def _capacity_holdup(interlocked: tuple[str, ...] = ()) -> VariableHoldup:
    return VariableHoldup(
        SCHEMA, ("feed",), ("pump",), "spill", overflow_law=OverflowLaw.CAPACITY, interlocked_draws=interlocked, context=CONTEXT
    )


class TestCapacityLaw:
    """The 9 -> 10 m3 example: capacity 10 m3, feed 2 m3/h, draw 1 m3/h, followed through the law at each mode."""

    FEED = {"feed": _stream(2.0, 3.0, 0.0)}

    def _rates(self, volume: float, active: int, draw: float = 1.0) -> tuple[float, float, np.ndarray]:
        unit = _capacity_holdup()
        state = HoldupState(jnp.asarray(volume), jnp.array([volume * 1.0, volume * 0.5]))
        inputs = _inputs(q_pump=draw) | {CAPACITY_ACTIVE: jnp.asarray(active)}
        spill = float(unit_outlets(unit, state, self.FEED, inputs, CAPACITY_PARAMS)["spill"].flow)
        derivative = unit.derivatives(state, self.FEED, inputs, CAPACITY_PARAMS)
        return spill, float(derivative.volume_m3), np.asarray(derivative.amount_coordinates)

    def test_below_capacity_nothing_spills_and_the_example_reaches_capacity_after_one_hour(self):
        """With the boundary mode inactive the volume rises at the full net inflow of 1 m3/h, so 9 m3 reaches 10 m3 at 1 h."""
        spill, volume_rate, _ = self._rates(9.0, active=0)
        assert spill == 0.0
        assert volume_rate == pytest.approx(1.0)
        assert (10.0 - 9.0) / volume_rate == pytest.approx(1.0)

    def test_at_capacity_the_net_inflow_spills_and_the_example_overflows_one_cubic_metre(self):
        """Active, the overflow takes max(b, 0) = 1 m3/h and holds V at 10 m3, so over the second hour 1 m3 spills."""
        spill, volume_rate, _ = self._rates(10.0, active=1)
        assert spill == pytest.approx(1.0)
        assert volume_rate == 0.0
        assert spill * (2.0 - 1.0) == pytest.approx(1.0)

    def test_overflow_removes_solute_at_the_tank_concentration_not_the_feed_concentration(self):
        """With feed and tank at unequal concentrations the spill carries the tank's: n_dot = Q_in C_in - (Q_draw + Q_spill) C."""
        _, _, amount_rate = self._rates(10.0, active=1)
        assert np.allclose(amount_rate, [2.0 * 3.0 - 2.0 * 1.0, 2.0 * 0.0 - 2.0 * 0.5])

    def test_negative_net_inflow_at_the_boundary_spills_nothing(self):
        """When draws exceed the feed the active law gives V_dot = min(b, 0): the tank drains and does not overflow."""
        spill, volume_rate, _ = self._rates(10.0, active=1, draw=3.0)
        assert spill == 0.0
        assert volume_rate == pytest.approx(-1.0)

    def test_overflow_flow_reads_inlet_flows_while_composition_stays_state_only(self):
        """The capacity group reads every inlet flow and the mode; composition still reads no inlet, so recycles stay acyclic."""
        groups = {group.name: group for group in _capacity_holdup().evaluation_groups}
        assert groups["capacity_overflow"].reads == (LocalField("feed", StreamField.FLOW),)
        assert CAPACITY_ACTIVE in groups["capacity_overflow"].input_names
        assert groups["composition"].reads == ()
        assert "weir_flow" not in groups

    def test_capacity_law_needs_an_overflow_outlet(self):
        """There is nothing to spill into without an overflow port."""
        with pytest.raises(ValueError, match="overflow outlet"):
            VariableHoldup(SCHEMA, ("feed",), ("pump",), overflow_law=OverflowLaw.CAPACITY, context=CONTEXT)


class TestWetInterlock:
    def test_interlock_shuts_only_the_declared_draws(self):
        """With the interlock mode set, interlocked draws stop and the others keep running."""
        unit = VariableHoldup(SCHEMA, ("feed",), ("recirc", "blowdown"), interlocked_draws=("blowdown",), context=CONTEXT)
        state = HoldupState(jnp.asarray(5.0), jnp.ones(2))
        feed = {"feed": _stream(0.0, 0.0, 0.0)}
        signals = _inputs(q_recirc=2.0, q_blowdown=1.0)
        running = unit_outlets(unit, state, feed, signals | {DRAW_INTERLOCK: jnp.asarray(0)}, PLAIN_PARAMS)
        tripped = unit_outlets(unit, state, feed, signals | {DRAW_INTERLOCK: jnp.asarray(1)}, PLAIN_PARAMS)
        assert (float(running["recirc"].flow), float(running["blowdown"].flow)) == (2.0, 1.0)
        assert (float(tripped["recirc"].flow), float(tripped["blowdown"].flow)) == (2.0, 0.0)

    def test_interlocked_draw_must_be_a_draw(self):
        """An interlock can only shut a declared withdrawal."""
        with pytest.raises(ValueError, match="interlocked draws"):
            VariableHoldup(SCHEMA, ("feed",), ("pump",), interlocked_draws=("other",), context=CONTEXT)


class TestHoldupPolicy:
    def _failed(self, unit: VariableHoldup, volume: float, coordinates: jax.Array) -> dict[str, bool]:
        result = holdup_policy(unit).check(HoldupState(jnp.asarray(volume), coordinates), {}, PLAIN_PARAMS, {})
        return dict(zip(HOLDUP_CHECKS, np.asarray(result.failed).tolist(), strict=True))

    def test_minimum_volume_is_admissible(self):
        """V = V_min is inside the wet domain, so the wet-level surface can be evaluated on the boundary."""
        assert not any(self._failed(_holdup(), 0.5, jnp.ones(2)).values())

    def test_volume_below_the_minimum_fails_only_the_continuation_check(self):
        """Just below V_min the physics is still evaluable; only the located wet boundary is violated."""
        failed = self._failed(_holdup(), 0.4, jnp.ones(2))
        assert failed == {"finite": False, "positive_volume": False, "reconstructable": False, "wet_volume": True}
        assert holdup_policy(_holdup()).continuation == ("wet_volume",)

    @pytest.mark.parametrize("volume", [0.0, -1.0])
    def test_empty_or_negative_volume_fails_the_hard_volume_check(self, volume: float):
        """A volume that cannot be divided by is a hard failure, checked before any stage evaluation."""
        assert self._failed(_holdup(), volume, jnp.ones(2))["positive_volume"]

    def test_nonfinite_coordinates_fail(self):
        """A NaN inventory is caught by the cheap state check."""
        assert self._failed(_holdup(), 5.0, jnp.array([jnp.nan, 1.0]))["finite"]

    def test_fast_chart_reconstruction_allows_roundoff_but_not_a_real_deficit(self):
        """A product negative by roundoff of its reconstruction is accepted unclipped; a material deficit is rejected."""
        _, closure, unit = TestFastChart()._unit()
        coordinates = closure.initialize(jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 1.0])).coordinates
        sulfur_row = len(closure.untouched) + 1
        roundoff = coordinates.at[sulfur_row].add(-1e-15)
        deficit = coordinates.at[sulfur_row].add(-1e-3)
        assert not self._failed(unit, 5.0, roundoff)["reconstructable"]
        assert self._failed(unit, 5.0, deficit)["reconstructable"]

    def test_witness_is_a_valid_wet_state(self):
        """The domain-safe placeholder for a failing lane passes every check of the policy."""
        policy = holdup_policy(_holdup())
        assert policy.witness is not None
        safe = policy.witness(HoldupState(jnp.asarray(-3.0), jnp.array([jnp.nan, 1.0])), PLAIN_PARAMS)
        assert not bool(jnp.any(policy.check(safe, {}, PLAIN_PARAMS, {}).failed))

    def test_events_follow_the_declared_laws(self):
        """A capacity holdup with an interlock and a fast chart declares entry, release, interlock, depletion and the I = 0 branch."""
        schema, closure, _ = TestFastChart()._unit()
        unit = VariableHoldup(
            schema, ("feed",), ("pump",), "spill", chart=closure, overflow_law=OverflowLaw.CAPACITY, interlocked_draws=("pump",), context=CONTEXT
        )
        policy = holdup_policy(unit)
        assert [event.name for event in policy.events] == ["capacity_entry", "capacity_release", "wet_interlock", "depletion", "fast_branch"]
        assert dict(policy.modes) == {CAPACITY_ACTIVE: CAPACITY_ACTIVE, DRAW_INTERLOCK: DRAW_INTERLOCK}
        terminal = [event.name for event in policy.events if event.continuation is Continuation.TERMINAL]
        assert terminal == ["depletion"]

    def test_event_surfaces_measure_their_physical_margins(self):
        """Entry measures headroom V_c - V, release the net inflow b, the wet events V - V_min, in their declared units."""
        policy = holdup_policy(_capacity_holdup(interlocked=("pump",)))
        events = {event.name: event for event in policy.events}
        state = HoldupState(jnp.asarray(9.0), jnp.ones(2))
        modes = {CAPACITY_ACTIVE: jnp.asarray(0), DRAW_INTERLOCK: jnp.asarray(0)}
        inputs = _inputs(q_pump=1.5) | modes
        fields = {LocalField("feed", StreamField.FLOW): jnp.asarray(2.0)}

        def surface(name: str) -> float:
            return float(events[name].surface(state, CAPACITY_PARAMS, modes, inputs, fields))

        assert surface("capacity_entry") == pytest.approx(1.0)
        assert surface("capacity_release") == pytest.approx(0.5)
        assert surface("wet_interlock") == surface("depletion") == pytest.approx(8.5)
        assert events["capacity_release"].stage_reads == (LocalField("feed", StreamField.FLOW),)

    def test_holdup_with_its_policy_builds_with_modes_on_their_ports(self):
        """The boundary modes are carried by the holdup's own input ports, outside the integrated state."""
        unit = _capacity_holdup()
        plant = build_flowsheet(
            {"sump": Node(unit, CAPACITY_PARAMS, policy=holdup_policy(unit))},
            [],
            {"feed": Port("sump", "feed")},
            {"pump": Port("sump", "pump"), "spill": Port("sump", "spill")},
        )
        assert plant.mode_ports == {f"sump.{CAPACITY_ACTIVE}": Port("sump", CAPACITY_ACTIVE)}
