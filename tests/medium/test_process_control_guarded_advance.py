"""The guarded advance on a variable holdup: capacity events, wet-boundary guards, transactional ledgers and lanes."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from process_control.binding import NamedAccess, Requirement
from process_control.chemistry.fast_closure import FastReactionClosure
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.fields import Phase, PrescribedContext
from process_control.flowsheet import Flowsheet, Node, Port, RhsEvaluation, build_flowsheet
from process_control.integration import advance_ode
from process_control.integration_records import AdvanceResult, AdvanceStatus, IntegrationContext
from process_control.inventory import HoldupState
from process_control.kinetics import KineticSource
from process_control.ledger import LedgerSpec, fast_extent_ledger
from process_control.streams import Stream
from process_control.units.holdup import CAPACITY_ACTIVE, DRAW_INTERLOCK, HoldupParams, OverflowLaw, VariableHoldup, holdup_policy

PROXY = ComponentBasis("mol/m3", Quantity.PROXY, "x")
CONTEXT = PrescribedContext(pressure_pa=101325.0)
TEMPERATURE = jnp.asarray(293.15)
V_MIN = 0.5


@pytest.fixture(scope="module", autouse=True)
def float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _schema(*names: str) -> ComponentSchema:
    n = len(names)
    return ComponentSchema(names, (False,) * n, (0.0,) * n, (PROXY,) * n)


SCHEMA = _schema("a", "b")


def _stream(flow: float, components: list[float], schema: ComponentSchema = SCHEMA) -> Stream:
    return Stream(jnp.asarray(flow), jnp.asarray(components), TEMPERATURE, jnp.asarray(101325.0), schema, Phase.LIQUID)


def _boundary_ledger(unit: VariableHoldup) -> LedgerSpec[Mapping[str, object], RhsEvaluation]:
    """Volume and amount flows across the sump boundary, read from the stage's resolved streams."""

    def rates(state: Mapping[str, object], evaluation: RhsEvaluation, inputs: Mapping[str, jax.Array], context: IntegrationContext):
        streams = {outlet: evaluation.streams[Port("sump", outlet)] for outlet in unit.outlets}
        flows = {f"{outlet}_m3": stream.flow for outlet, stream in streams.items()}
        removed = sum((stream.flow * stream.components for stream in streams.values()), jnp.zeros(len(unit.schema.names)))
        return {**flows, "removed": removed}

    units = {**{f"{outlet}_m3": "m3/h" for outlet in unit.outlets}, "removed": "mol/h"}
    return LedgerSpec("boundary", units, rates)


def _sump_plant(unit: VariableHoldup, params: HoldupParams) -> Flowsheet:
    return build_flowsheet(
        {"sump": Node(unit, params, policy=holdup_policy(unit))},
        [],
        {inlet: Port("sump", inlet) for inlet in unit.inlets},
        {outlet: Port("sump", outlet) for outlet in unit.outlets},
        ledgers=(_boundary_ledger(unit),),
    )


def _volume(result: AdvanceResult[Mapping[str, object]]) -> float:
    state = result.state["sump"]
    assert isinstance(state, HoldupState)
    return float(state.volume_m3)


def _amounts(result: AdvanceResult[Mapping[str, object]]) -> np.ndarray:
    state = result.state["sump"]
    assert isinstance(state, HoldupState)
    return np.asarray(state.amount_coordinates)


class TestFixedCapacity:
    """Start at 9 m3 under a 10 m3 capacity, feed 2 m3/h at 3 mol/m3, draw 1 m3/h, for two hours."""

    FEED_A = 3.0
    START_A = 1.0

    def _advance(self) -> tuple[Flowsheet, AdvanceResult[Mapping[str, object]]]:
        unit = VariableHoldup(SCHEMA, ("feed",), ("pump",), "spill", overflow_law=OverflowLaw.CAPACITY, context=CONTEXT)
        plant = _sump_plant(unit, HoldupParams(v_min_m3=V_MIN, capacity_m3=10.0))
        state = {"sump": HoldupState(jnp.asarray(9.0), jnp.array([9.0 * self.START_A, 0.0]))}
        inputs = {"temperature_k": TEMPERATURE, "q_pump": jnp.asarray(1.0)}
        context = plant.initial_context({f"sump.{CAPACITY_ACTIVE}": 0})
        return plant, plant.advance(state, {"feed": _stream(2.0, [self.FEED_A, 0.0])}, inputs, 2.0, 8, context)

    def test_capacity_is_reached_and_held_without_clipping(self):
        """The advance completes, enters the boundary mode, and ends at the capacity with no snapped volume beyond tolerance."""
        _, result = self._advance()

        assert int(result.status) == AdvanceStatus.SUCCESS
        assert float(result.completed_time_h) == 2.0
        assert int(result.context.modes[f"sump.{CAPACITY_ACTIVE}"]) == 1
        assert int(result.checks.event_index) == 0
        assert _volume(result) == pytest.approx(10.0, abs=1e-8)

    def test_draw_and_overflow_integrals_match_the_piecewise_solution(self):
        """Drawing 1 m3/h for 2 h removes 2 m3; overflow starts at the 1 h event and spills 1 m3."""
        _, result = self._advance()
        boundary = result.quadratures["boundary"]

        assert float(boundary["pump_m3"]) == pytest.approx(2.0, abs=1e-9)
        assert float(boundary["spill_m3"]) == pytest.approx(1.0, abs=1e-6)

    def test_solute_leaves_at_the_tank_concentration_through_both_outlets(self):
        """With feed and tank at unequal concentrations, inventory and removed solute follow the analytic two-phase solution."""
        _, result = self._advance()
        n0, c_in = 9.0 * self.START_A, self.FEED_A
        n1 = (9.0 * n0 + c_in * (10.0**2 - 9.0**2)) / 10.0
        n2 = 10.0 * c_in + (n1 - 10.0 * c_in) * np.exp(-0.2)

        assert _amounts(result)[0] == pytest.approx(n2, rel=1e-7)
        removed = float(result.quadratures["boundary"]["removed"][0])
        assert removed == pytest.approx(n0 + 2.0 * c_in * 2.0 - n2, rel=1e-7)


class TestWetBoundary:
    def _drain(
        self, interlocked: tuple[str, ...], volume: float = 2.0, draw: float | jax.Array = 3.0, recorder: "ConcentrationRecorder | None" = None
    ) -> AdvanceResult[Mapping[str, object]]:
        sources = (KineticSource.bind(recorder, None, SCHEMA),) if recorder is not None else ()
        unit = VariableHoldup(SCHEMA, ("feed",), ("pump",), sources=sources, interlocked_draws=interlocked, context=CONTEXT)
        plant = _sump_plant(unit, HoldupParams(v_min_m3=V_MIN))
        state = {"sump": HoldupState(jnp.asarray(volume), jnp.array([2.0 * volume, volume]))}
        modes = {f"sump.{DRAW_INTERLOCK}": 0} if interlocked else {}
        inputs = {"temperature_k": TEMPERATURE, "q_pump": jnp.asarray(draw)}
        return plant.advance(state, {"feed": _stream(0.0, [0.0, 0.0])}, inputs, 1.0, 1, plant.initial_context(modes))

    def test_oversized_step_stops_at_the_wet_boundary_as_depleted(self):
        """Draining 3 m3/h from 2 m3 over one 1 h step reaches V_min at 0.5 h: the advance stops there, not below it."""
        result = self._drain(())

        assert int(result.status) == AdvanceStatus.DEPLETED
        assert float(result.completed_time_h) == pytest.approx(0.5, abs=1e-8)
        assert V_MIN <= _volume(result) <= V_MIN + 1e-8
        assert np.allclose(_amounts(result) / _volume(result), [2.0, 1.0])

    def test_chemistry_is_never_evaluated_at_a_negative_volume(self):
        """Stages of the oversized trial that would hold negative volume are rejected before any source sees their concentrations."""
        recorder = ConcentrationRecorder()

        self._drain((), recorder=recorder)

        seen = np.concatenate(recorder.seen)
        assert seen.size > 0
        assert np.all(np.isfinite(seen)) and np.all(seen >= 0.0)

    def test_rejected_trials_contribute_nothing_to_the_ledger(self):
        """After rejections and an event split, the drawn volume equals the volume actually lost, so no rejected trial leaked in."""
        result = self._drain(())

        assert int(result.checks.retries) > 0
        assert float(result.quadratures["boundary"]["pump_m3"]) == pytest.approx(2.0 - _volume(result), abs=1e-12)

    def test_interlock_shuts_the_draw_and_the_advance_continues(self):
        """With the draw interlocked, reaching V_min trips it and the holdup rests there for the rest of the step."""
        result = self._drain(("pump",))

        assert int(result.status) == AdvanceStatus.SUCCESS
        assert float(result.completed_time_h) == 1.0
        assert int(result.context.modes[f"sump.{DRAW_INTERLOCK}"]) == 1
        assert V_MIN <= _volume(result) <= V_MIN + 1e-8

    def test_invalid_initial_state_completes_no_time(self):
        """A state already outside the wet domain is returned untouched with zero completed time."""
        result = self._drain((), volume=0.4)

        assert int(result.status) == AdvanceStatus.INVALID_INITIAL_STATE
        assert float(result.completed_time_h) == 0.0
        assert _volume(result) == 0.4

    def test_failing_lane_does_not_disturb_a_successful_lane(self):
        """Under jit and vmap each lane matches its own scalar advance, a depleting lane beside a successful one."""

        def run(draw: jax.Array) -> AdvanceResult[Mapping[str, object]]:
            return self._drain((), draw=draw)

        batched = jax.jit(jax.vmap(run))(jnp.array([1.0, 3.0]))

        for lane, draw in enumerate((1.0, 3.0)):
            single = self._drain((), draw=draw)
            assert int(batched.status[lane]) == int(single.status)
            assert float(batched.completed_time_h[lane]) == pytest.approx(float(single.completed_time_h), abs=1e-12)
            lane_state = batched.state["sump"]
            assert isinstance(lane_state, HoldupState)
            assert np.allclose(lane_state.amount_coordinates[lane], _amounts(single), atol=1e-12)
        assert int(batched.status[0]) == AdvanceStatus.SUCCESS
        assert int(batched.status[1]) == AdvanceStatus.DEPLETED


@dataclass(frozen=True, eq=False)
class ConcentrationRecorder:
    """A zero-rate source that records every concentration it is evaluated on."""

    seen: list[np.ndarray] = field(default_factory=list)

    @property
    def access(self) -> NamedAccess:
        return NamedAccess(reads=(Requirement("a", PROXY), Requirement("b", PROXY)), writes=(Requirement("a", PROXY),))

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: None) -> jax.Array:
        jax.debug.callback(self._record, reads)
        return jnp.zeros(1, dtype=reads.dtype)

    def _record(self, values: jax.Array) -> None:
        self.seen.append(np.asarray(values))


class TestFastChartCrossing:
    NAMES = ("sulfide", "chlorine", "sulfur", "sulfate", "chloride", "sodium")
    YIELD = 0.5

    def test_extent_ledger_closes_across_the_branch_change(self):
        """Sulfide feed into a chlorine-excess sump drives I through zero; the accepted ledger explains every inventory row."""
        d = 1.0 + 3.0 * self.YIELD
        stoichiometry = np.array([-1.0, -d, 1.0 - self.YIELD, self.YIELD, d, 0.0])
        schema = _schema(*self.NAMES)
        closure = FastReactionClosure(schema, stoichiometry, reactant=0, oxidant=1, oxidant_products=(4,))
        unit = VariableHoldup(schema, ("feed",), ("pump",), chart=closure, context=CONTEXT)
        feed = _stream(1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], schema)
        plant = build_flowsheet(
            {"sump": Node(unit, HoldupParams(v_min_m3=V_MIN), policy=holdup_policy(unit))},
            [],
            {"feed": Port("sump", "feed")},
            {"pump": Port("sump", "pump")},
            ledgers=(_nonfast_ledger(feed),),
        )
        start = closure.initialize(jnp.array([0.0, 1.0, 0.0, 0.0, 0.5, 3.0]))
        state = {"sump": HoldupState(jnp.asarray(5.0), start.coordinates)}
        inputs = {"temperature_k": TEMPERATURE, "q_pump": jnp.asarray(1.0)}

        result = plant.advance(state, {"feed": feed}, inputs, 2.0, 4, plant.initial_context({}))

        assert int(result.status) == AdvanceStatus.SUCCESS
        invariant = len(closure.untouched)
        assert float(start.coordinates[invariant]) > 0.0 > float(_amounts(result)[invariant])
        assert int(result.checks.event_index) == [event.name for event in plant.nodes["sump"].policy.events].index("fast_branch")
        ledger = fast_extent_ledger(
            start.amounts,
            closure.amounts(jnp.asarray(_amounts(result))),
            result.quadratures["nonfast"]["amount"],
            jnp.asarray(stoichiometry),
            row=0,
            rtol=1e-9,
            atol=1e-12,
        )
        assert bool(ledger.ok)


def _nonfast_ledger(feed: Stream) -> LedgerSpec[Mapping[str, object], RhsEvaluation]:
    """Physical transport into and out of the sump, the only non-fast amount rate when no slow source acts."""

    def rates(state: Mapping[str, object], evaluation: RhsEvaluation, inputs: Mapping[str, jax.Array], context: IntegrationContext):
        pump = evaluation.streams[Port("sump", "pump")]
        return {"amount": feed.flow * feed.components - pump.flow * pump.components}

    return LedgerSpec("nonfast", {"amount": "mol/h"}, rates)


class TestBareDerivative:
    def test_exhausted_retries_return_only_accepted_progress(self):
        """A rate undefined from x = 1 on cannot be crossed: halving stalls short of it and the advance reports how far it got."""

        def rate(x: jax.Array) -> jax.Array:
            return jnp.where(x < 1.0, 1.0, jnp.nan)

        result = advance_ode(rate, _unprojected, jnp.asarray(0.0), 2.0, 1)

        assert int(result.status) == AdvanceStatus.RETRY_EXHAUSTED
        assert 0.0 < float(result.completed_time_h) < 1.0
        assert float(result.state) == pytest.approx(float(result.completed_time_h), abs=1e-12)

    def test_ordinary_substeps_complete_exactly_dt(self):
        """Without rejections the completed time is the requested dt itself, not a sum of substeps."""
        result = advance_ode(_unit_decay, _unprojected, jnp.asarray(1.0), 0.3, 7)

        assert int(result.status) == AdvanceStatus.SUCCESS
        assert float(result.completed_time_h) == 0.3
        assert float(result.state) == pytest.approx(np.exp(-0.3), rel=1e-8)


def _unit_decay(x: jax.Array) -> jax.Array:
    return -x


def _unprojected(x: jax.Array) -> jax.Array:
    return x
