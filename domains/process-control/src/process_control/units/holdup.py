"""Variable-volume, well-mixed liquid holdup with prescribed draws and an optional overflow.

State is the wet volume plus the chart's amount coordinates. All outlets carry the one well-mixed
concentration, so composition depends on state alone and no outlet reads an inlet; that is what lets a
recirculation loop through the holdup compile as an acyclic field graph. The overflow follows either a
finite weir or a fixed-capacity law whose boundary mode is a declared input, never a state leaf.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.fast_closure import FastReactionClosure
from process_control.components import ComponentSchema
from process_control.events import Continuation, CrossingDirection, EventSpec, EventTolerance, Modes
from process_control.fields import CONTEXT_TEMPERATURE, EvaluationGroup, GroupResult, LocalField, Phase, PortSpec, PrescribedContext, StreamField
from process_control.integration_policy import IntegrationPolicy
from process_control.integration_records import CheckResult, check_result
from process_control.inventory import HoldupState, IdentityChart, InventoryChart, inventory_balance, reconstruct
from process_control.streams import Stream
from process_control.units.hydraulics import LevelGeometry, RectangularWeir, weir_overflow_m3_per_h
from process_control.units.reactive_cstr import Source

CAPACITY_ACTIVE = "capacity_active"
DRAW_INTERLOCK = "draw_interlock"


class OverflowLaw(Enum):
    """A weir discharging the head over its crest, or a rigid capacity spilling whatever cannot be stored."""

    WEIR = "weir"
    CAPACITY = "capacity"


@jax_dataclass
class HoldupParams:
    """The wet-domain minimum volume (m³) and the overflow's weir with its level geometry, or its fixed capacity (m³)."""

    v_min_m3: jax.Array | float
    geometry: LevelGeometry | None = None
    weir: RectangularWeir | None = None
    capacity_m3: jax.Array | float | None = None


class Hydraulics(NamedTuple):
    """Realized outlet flows in m³/h: one per draw in declared order, and the overflow."""

    draws: jax.Array
    overflow: jax.Array


def draw_signal(draw: str) -> str:
    """The plant signal (m³/h) prescribing the named draw."""
    return f"q_{draw}"


@dataclass(frozen=True)
class VariableHoldup:
    """Well-mixed liquid inventory fed by ordered `inlets`, drained by prescribed `draws` and an optional `overflow`.

    Draw flows come from their `q_<draw>` signals alone, without reading an inlet or capping against the
    contents, so a draw above the feed drains storage; a wet-level interlock mode shuts the
    `interlocked_draws`. Under the weir law the overflow follows from the stored volume over the weir in
    `HoldupParams`, a head over the crest rather than a hard capacity. Under the capacity law it is zero
    until the `capacity_active` mode holds, then spills the positive net inflow so the volume stays at the
    boundary; reaching and leaving the boundary are events of `holdup_policy`. Every bound source
    is evaluated on the reconstructed full physical concentration and its rate multiplied by the volume
    once. The chart stores either the full inventories or an invariant reduction. Outlets leave at the
    prescribed `context`, whose Kelvin temperature signal is also what the sources receive.
    """

    schema: ComponentSchema
    inlets: tuple[str, ...]
    draws: tuple[str, ...]
    overflow: str | None = None
    sources: tuple[Source, ...] = ()
    chart: InventoryChart = field(default_factory=IdentityChart)
    overflow_law: OverflowLaw = OverflowLaw.WEIR
    interlocked_draws: tuple[str, ...] = ()
    context: PrescribedContext = field(kw_only=True)

    def __post_init__(self) -> None:
        if not self.inlets:
            raise ValueError("a holdup needs at least one inlet")
        outlets = self.outlets
        if not outlets:
            raise ValueError("a holdup needs at least one draw or an overflow")
        if len(set(self.inlets)) != len(self.inlets) or len(set(outlets)) != len(outlets):
            raise ValueError(f"inlet names {self.inlets} and outlet names {outlets} must each be unique")
        for source in self.sources:
            if source.schema != self.schema:
                raise ValueError(f"a source bound to {source.schema.names} cannot act on a holdup carrying {self.schema.names}")
        if not set(self.interlocked_draws) <= set(self.draws):
            raise ValueError(f"interlocked draws {self.interlocked_draws} must be among the draws {self.draws}")
        if self.overflow_law is OverflowLaw.CAPACITY and self.overflow is None:
            raise ValueError("a fixed-capacity law needs an overflow outlet")

    @property
    def outlets(self) -> tuple[str, ...]:
        return (*self.draws, *((self.overflow,) if self.overflow is not None else ()))

    @property
    def inlet_ports(self) -> Mapping[str, PortSpec]:
        return dict.fromkeys(self.inlets, PortSpec(self.schema, Phase.LIQUID))

    @property
    def outlet_ports(self) -> Mapping[str, PortSpec]:
        return dict.fromkeys(self.outlets, PortSpec(self.schema, Phase.LIQUID))

    @property
    def evaluation_groups(self) -> tuple[EvaluationGroup, ...]:
        groups: list[EvaluationGroup] = []
        if self.draws:
            groups.append(
                EvaluationGroup(
                    "draw_flows",
                    outputs=tuple(LocalField(draw, StreamField.FLOW) for draw in self.draws),
                    reads=(),
                    input_names=self._draw_inputs,
                )
            )
        if self.overflow is not None and self.overflow_law is OverflowLaw.WEIR:
            groups.append(EvaluationGroup("weir_flow", outputs=(LocalField(self.overflow, StreamField.FLOW),), reads=(), reads_state=True))
        if self.overflow is not None and self.overflow_law is OverflowLaw.CAPACITY:
            groups.append(
                EvaluationGroup(
                    "capacity_overflow",
                    outputs=(LocalField(self.overflow, StreamField.FLOW),),
                    reads=tuple(LocalField(inlet, StreamField.FLOW) for inlet in self.inlets),
                    input_names=(*self._draw_inputs, CAPACITY_ACTIVE),
                )
            )
        groups.append(
            EvaluationGroup(
                "composition", outputs=tuple(LocalField(outlet, StreamField.COMPONENTS) for outlet in self.outlets), reads=(), reads_state=True
            )
        )
        groups.append(self.context.group(self.outlets))
        return tuple(groups)

    @property
    def input_ports(self) -> tuple[str, ...]:
        signals = tuple(dict.fromkeys(signal for source in self.sources for signal in source.signals))
        return (*self._draw_inputs, *signals, CONTEXT_TEMPERATURE, *self._mode_ports)

    @property
    def stateful(self) -> bool:
        return True

    def initial_state(self, volume_m3: float, amounts: jax.Array) -> HoldupState:
        """State holding `volume_m3` and the physical inventories `amounts` in the chart's coordinates."""
        return HoldupState(jnp.asarray(volume_m3), self.chart.coordinates(amounts))

    @property
    def mode_ports(self) -> dict[str, str]:
        """Each boundary mode this holdup reads, mapped to the input port carrying it."""
        return {mode: mode for mode in self._mode_ports}

    def hydraulics(self, state: HoldupState, inlet_flows: jax.Array, inputs: Mapping[str, jax.Array], params: HoldupParams) -> Hydraulics:
        """Realized draw and overflow flows; groups and derivatives share it. Only the capacity law reads `inlet_flows`."""
        draws = self._draw_flows(inputs)
        return Hydraulics(draws, self._overflow_flow(state, inlet_flows, draws, inputs, params))

    def net_inflow(self, inlet_flows: jax.Array, inputs: Mapping[str, jax.Array]) -> jax.Array:
        """Unconstrained net balance b (m³/h): inflow less realized draws, before any overflow."""
        return jnp.sum(inlet_flows) - jnp.sum(self._draw_flows(inputs))

    def evaluate_group(
        self, group_name: str, state: HoldupState, reads: Mapping[LocalField, jax.Array], inputs: Mapping[str, jax.Array], params: HoldupParams
    ) -> GroupResult:
        match group_name:
            case "draw_flows":
                return GroupResult(tuple(self._draw_flows(inputs)))
            case "weir_flow":
                return GroupResult((self._weir_flow(state, params),))
            case "capacity_overflow":
                inlet_flows = jnp.stack([reads[LocalField(inlet, StreamField.FLOW)] for inlet in self.inlets])
                return GroupResult((self._capacity_flow(inlet_flows, self._draw_flows(inputs), inputs),))
            case "composition":
                concentrations = reconstruct(self.chart, state).concentrations
                return GroupResult((concentrations,) * len(self.outlets))
            case "context":
                return self.context.evaluate(self.outlets, inputs)
        raise ValueError(f"variable holdup has no group {group_name!r}")

    def derivatives(self, state: HoldupState, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: HoldupParams) -> HoldupState:
        inventory = reconstruct(self.chart, state)
        inlet_flows = jnp.stack([inlets[name].flow for name in self.inlets])
        hydraulics = self.hydraulics(state, inlet_flows, inputs, params)

        source_rate = jnp.zeros_like(inventory.concentrations)
        for source in self.sources:
            signals = {signal: inputs[signal] for signal in source.signals}
            source_rate = source.add_to(source_rate, inventory.concentrations, signals, inputs[CONTEXT_TEMPERATURE])

        balance = inventory_balance(
            self.chart,
            state,
            inventory,
            inlet_flows,
            jnp.stack([inlets[name].components for name in self.inlets]),
            hydraulics.draws,
            hydraulics.overflow,
            source_rate,
        )
        return HoldupState(balance.volume_rate, balance.coordinate_rate)

    def project(self, state: HoldupState, params: HoldupParams) -> HoldupState:
        return state

    @property
    def _draw_inputs(self) -> tuple[str, ...]:
        return (*(draw_signal(draw) for draw in self.draws), *((DRAW_INTERLOCK,) if self.interlocked_draws else ()))

    @property
    def _mode_ports(self) -> tuple[str, ...]:
        capacity = (CAPACITY_ACTIVE,) if self.overflow_law is OverflowLaw.CAPACITY else ()
        return (*capacity, *((DRAW_INTERLOCK,) if self.interlocked_draws else ()))

    def _draw_flows(self, inputs: Mapping[str, jax.Array]) -> jax.Array:
        if not self.draws:
            return jnp.zeros(0)
        flows = [inputs[draw_signal(draw)] for draw in self.draws]
        if self.interlocked_draws:
            running = inputs[DRAW_INTERLOCK] == 0
            flows = [jnp.where(running, flow, 0.0) if draw in self.interlocked_draws else flow for draw, flow in zip(self.draws, flows, strict=True)]
        return jnp.stack(flows)

    def _overflow_flow(
        self, state: HoldupState, inlet_flows: jax.Array, draws: jax.Array, inputs: Mapping[str, jax.Array], params: HoldupParams
    ) -> jax.Array:
        if self.overflow is None:
            return jnp.zeros(())
        if self.overflow_law is OverflowLaw.CAPACITY:
            return self._capacity_flow(inlet_flows, draws, inputs)
        return self._weir_flow(state, params)

    def _weir_flow(self, state: HoldupState, params: HoldupParams) -> jax.Array:
        if params.geometry is None or params.weir is None:
            raise ValueError("a holdup with a weir overflow needs a level geometry and a weir in its params")
        return weir_overflow_m3_per_h(state.volume_m3, params.geometry, params.weir)

    def _capacity_flow(self, inlet_flows: jax.Array, draws: jax.Array, inputs: Mapping[str, jax.Array]) -> jax.Array:
        """At the boundary Q_overflow = max(b, 0), so V_dot = min(b, 0); below it nothing spills."""
        net = jnp.sum(inlet_flows) - jnp.sum(draws)
        return jnp.where(inputs[CAPACITY_ACTIVE] != 0, jnp.maximum(net, 0.0), 0.0)


@dataclass(frozen=True)
class HoldupTolerance:
    """Event tolerances in physical units: time (h), volume (m³), net flow (m³/h) and chart invariant amount."""

    time_h: float = 1e-9
    volume_m3: float = 1e-9
    flow_m3_per_h: float = 1e-9
    amount: float = 1e-9


DEFAULT_TOLERANCE = HoldupTolerance()
HOLDUP_CHECKS = ("finite", "positive_volume", "reconstructable", "wet_volume")


def holdup_policy(unit: VariableHoldup, tolerance: HoldupTolerance = DEFAULT_TOLERANCE) -> IntegrationPolicy[HoldupState, HoldupParams]:
    """The holdup's wet-domain checks, boundary modes and events.

    Volume and coordinates must be finite, the volume positive and the chart reconstructable within its
    roundoff tolerance; those are hard limits evaluated before any division. `V >= V_min` (admissible at
    equality) bounds the wet domain: it is a continuation check, located by the wet-level event, which
    shuts the interlocked draws or, with none or once they are shut, ends the advance as depleted. A
    capacity law enters its boundary mode when `V_c - V` falls through zero and leaves it when the net
    inflow turns negative. A fast chart segments trials where its signed invariant `I` changes sign.
    """
    modes = unit.mode_ports
    volume = EventTolerance(tolerance.time_h, tolerance.volume_m3)

    def check(state: HoldupState, inputs: Mapping[str, jax.Array], params: HoldupParams, current: Modes) -> CheckResult:
        v = state.volume_m3
        finite = jnp.isfinite(v) & jnp.all(jnp.isfinite(state.amount_coordinates))
        return check_result((~finite, ~(v > 0), ~unit.chart.reconstructable(state.amount_coordinates), ~(v >= params.v_min_m3)))

    def witness(state: HoldupState, params: HoldupParams) -> HoldupState:
        return HoldupState(jnp.full_like(state.volume_m3, params.v_min_m3), jnp.zeros_like(state.amount_coordinates))

    def wet_margin(state: HoldupState, params: HoldupParams, current: Modes, inputs: Modes, fields: Mapping[LocalField, jax.Array]) -> jax.Array:
        return state.volume_m3 - params.v_min_m3

    events: list[EventSpec[HoldupState, HoldupParams]] = []
    if unit.overflow_law is OverflowLaw.CAPACITY:
        events += _capacity_events(unit, tolerance)
    if unit.interlocked_draws:
        events.append(
            EventSpec("wet_interlock", wet_margin, "m3", CrossingDirection.DOWN, _mode_is(DRAW_INTERLOCK, 0), volume, _set(DRAW_INTERLOCK, 1), (DRAW_INTERLOCK,))
        )
    depleting = _mode_is(DRAW_INTERLOCK, 1) if unit.interlocked_draws else _always
    events.append(EventSpec("depletion", wet_margin, "m3", CrossingDirection.DOWN, depleting, volume, _no_change, (), continuation=Continuation.TERMINAL))
    if isinstance(unit.chart, FastReactionClosure):
        events.append(_branch_event(unit.chart, tolerance))

    return IntegrationPolicy(HOLDUP_CHECKS, check, modes=modes, events=tuple(events), continuation=("wet_volume",), witness=witness)


def _capacity_events(unit: VariableHoldup, tolerance: HoldupTolerance) -> list[EventSpec[HoldupState, HoldupParams]]:
    inflows = tuple(LocalField(inlet, StreamField.FLOW) for inlet in unit.inlets)

    def headroom(state: HoldupState, params: HoldupParams, current: Modes, inputs: Modes, fields: Mapping[LocalField, jax.Array]) -> jax.Array:
        if params.capacity_m3 is None:
            raise ValueError("a fixed-capacity holdup needs capacity_m3 in its params")
        return params.capacity_m3 - state.volume_m3

    def net_inflow(state: HoldupState, params: HoldupParams, current: Modes, inputs: Modes, fields: Mapping[LocalField, jax.Array]) -> jax.Array:
        return unit.net_inflow(jnp.stack([fields[read] for read in inflows]), inputs)

    return [
        EventSpec(
            "capacity_entry",
            headroom,
            "m3",
            CrossingDirection.DOWN,
            _mode_is(CAPACITY_ACTIVE, 0),
            EventTolerance(tolerance.time_h, tolerance.volume_m3),
            _set(CAPACITY_ACTIVE, 1),
            (CAPACITY_ACTIVE,),
        ),
        EventSpec(
            "capacity_release",
            net_inflow,
            "m3/h",
            CrossingDirection.DOWN,
            _mode_is(CAPACITY_ACTIVE, 1),
            EventTolerance(tolerance.time_h, tolerance.flow_m3_per_h),
            _set(CAPACITY_ACTIVE, 0),
            (CAPACITY_ACTIVE,),
            stage_reads=inflows,
        ),
    ]


def _branch_event(chart: FastReactionClosure, tolerance: HoldupTolerance) -> EventSpec[HoldupState, HoldupParams]:
    index = len(chart.untouched)

    def invariant(state: HoldupState, params: HoldupParams, current: Modes, inputs: Modes, fields: Mapping[LocalField, jax.Array]) -> jax.Array:
        return state.amount_coordinates[index]

    return EventSpec("fast_branch", invariant, "amount", CrossingDirection.BOTH, _always, EventTolerance(tolerance.time_h, tolerance.amount), _no_change, ())


def _mode_is(mode: str, value: int) -> Callable[[Modes], jax.Array]:
    def active(current: Modes) -> jax.Array:
        return current[mode] == value

    return active


def _set(mode: str, value: int) -> Callable[[Modes], Modes]:
    def transition(current: Modes) -> Modes:
        return {mode: jnp.asarray(value)}

    return transition


def _always(current: Modes) -> jax.Array:
    return jnp.ones((), bool)


def _no_change(current: Modes) -> Modes:
    return {}
