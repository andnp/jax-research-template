"""Variable-volume, well-mixed liquid holdup with prescribed draws and an optional weir overflow.

State is the wet volume plus the chart's amount coordinates. All outlets carry the one well-mixed
concentration, so composition depends on state alone and no outlet reads an inlet; that is what lets a
recirculation loop through the holdup compile as an acyclic field graph.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.components import ComponentSchema
from process_control.fields import CONTEXT_TEMPERATURE, EvaluationGroup, GroupResult, LocalField, Phase, PortSpec, PrescribedContext, StreamField
from process_control.inventory import HoldupState, IdentityChart, InventoryChart, inventory_balance, reconstruct
from process_control.streams import Stream
from process_control.units.hydraulics import LevelGeometry, RectangularWeir, weir_overflow_m3_per_h
from process_control.units.reactive_cstr import Source


@jax_dataclass
class HoldupParams:
    """The wet-domain minimum volume (m³) and, for a holdup with an overflow, its level geometry and weir."""

    v_min_m3: jax.Array | float
    geometry: LevelGeometry | None = None
    weir: RectangularWeir | None = None


class Hydraulics(NamedTuple):
    """Realized outlet flows in m³/h: one per draw in declared order, and the overflow."""

    draws: jax.Array
    overflow: jax.Array


def draw_signal(draw: str) -> str:
    """The plant signal (m³/h) prescribing the named draw."""
    return f"q_{draw}"


@dataclass(frozen=True)
class VariableHoldup:
    """Well-mixed liquid inventory fed by ordered `inlets`, drained by prescribed `draws` and an optional weir `overflow`.

    Draw flows come from their `q_<draw>` signals alone, without reading an inlet or capping against the
    contents, so a draw above the feed drains storage. The overflow flow follows from the stored volume
    over the weir in `HoldupParams`; it is a head over the crest, not a hard capacity. Every bound source
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
                    input_names=tuple(draw_signal(draw) for draw in self.draws),
                )
            )
        if self.overflow is not None:
            groups.append(EvaluationGroup("weir_flow", outputs=(LocalField(self.overflow, StreamField.FLOW),), reads=(), reads_state=True))
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
        return (*(draw_signal(draw) for draw in self.draws), *signals, CONTEXT_TEMPERATURE)

    @property
    def stateful(self) -> bool:
        return True

    def initial_state(self, volume_m3: float, amounts: jax.Array) -> HoldupState:
        """State holding `volume_m3` and the physical inventories `amounts` in the chart's coordinates."""
        return HoldupState(jnp.asarray(volume_m3), self.chart.coordinates(amounts))

    def hydraulics(self, state: HoldupState, inputs: Mapping[str, jax.Array], params: HoldupParams) -> Hydraulics:
        """Realized draw and overflow flows from signals and stored volume only; groups and derivatives share it."""
        return Hydraulics(self._draw_flows(inputs), self._overflow_flow(state, params))

    def evaluate_group(
        self, group_name: str, state: HoldupState, reads: Mapping[LocalField, jax.Array], inputs: Mapping[str, jax.Array], params: HoldupParams
    ) -> GroupResult:
        match group_name:
            case "draw_flows":
                return GroupResult(tuple(self._draw_flows(inputs)))
            case "weir_flow":
                return GroupResult((self._overflow_flow(state, params),))
            case "composition":
                concentrations = reconstruct(self.chart, state, params.v_min_m3).concentrations
                return GroupResult((concentrations,) * len(self.outlets))
            case "context":
                return self.context.evaluate(self.outlets, inputs)
        raise ValueError(f"variable holdup has no group {group_name!r}")

    def derivatives(self, state: HoldupState, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: HoldupParams) -> HoldupState:
        inventory = reconstruct(self.chart, state, params.v_min_m3)
        hydraulics = self.hydraulics(state, inputs, params)

        source_rate = jnp.zeros_like(inventory.concentrations)
        for source in self.sources:
            signals = {signal: inputs[signal] for signal in source.signals}
            source_rate = source.add_to(source_rate, inventory.concentrations, signals, inputs[CONTEXT_TEMPERATURE])

        balance = inventory_balance(
            self.chart,
            state,
            inventory,
            jnp.stack([inlets[name].flow for name in self.inlets]),
            jnp.stack([inlets[name].components for name in self.inlets]),
            hydraulics.draws,
            hydraulics.overflow,
            source_rate,
        )
        return HoldupState(balance.volume_rate, balance.coordinate_rate)

    def project(self, state: HoldupState, params: HoldupParams) -> HoldupState:
        return state

    def _draw_flows(self, inputs: Mapping[str, jax.Array]) -> jax.Array:
        return jnp.stack([inputs[draw_signal(draw)] for draw in self.draws]) if self.draws else jnp.zeros(0)

    def _overflow_flow(self, state: HoldupState, params: HoldupParams) -> jax.Array:
        if self.overflow is None:
            return jnp.zeros(())
        if params.geometry is None or params.weir is None:
            raise ValueError("a holdup with an overflow needs a level geometry and a weir in its params")
        return weir_overflow_m3_per_h(state.volume_m3, params.geometry, params.weir)
