from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.components import ComponentSchema
from process_control.kinetics import ZERO_CELSIUS
from process_control.streams import Stream
from process_control.units.contract import Outlet


class Source(Protocol):
    """A source term bound to a carrier schema that adds its rates, per hour, to an accumulated derivative."""

    @property
    def schema(self) -> ComponentSchema: ...

    @property
    def signals(self) -> tuple[str, ...]: ...

    def add_to(self, total: jax.Array, state: jax.Array, signals: Mapping[str, jax.Array], temperature_k: jax.Array) -> jax.Array: ...


def unchanged(state: jax.Array):
    return state


def nonnegative(state: jax.Array):
    return jnp.maximum(state, 0.0)


@jax_dataclass
class ReactiveCSTRParams:
    """Reactor volume (m³) and the temperature (°C) used when the reactor reads no temperature input."""

    volume: float
    temperature: float = 15.0


@dataclass(frozen=True)
class ReactiveCSTR:
    """Constant-volume CSTR whose composition changes by dilution plus the rates of its bound sources.

    Every source is evaluated on the same state and their rates add. The outlet passes the inlet flow
    through and carries the reactor state as its composition. Inputs are the sources' declared signals
    and, when `temperature_input` is set, `temperature` (°C), passed to sources in K. `projection` maps
    each integration substep's state onto the states the plant admits; it changes nothing by default.
    """

    schema: ComponentSchema
    sources: tuple[Source, ...] = ()
    temperature_input: bool = True
    projection: Callable[[jax.Array], jax.Array] = unchanged

    def __post_init__(self) -> None:
        for source in self.sources:
            if source.schema != self.schema:
                raise ValueError(f"a source bound to {source.schema.names} cannot act on a reactor carrying {self.schema.names}")

    @property
    def inlet_ports(self) -> Mapping[str, ComponentSchema]:
        return {"in": self.schema}

    @property
    def outlet_ports(self) -> Mapping[str, Outlet]:
        return {"out": Outlet(self.schema, flow_from=frozenset({"in"}))}

    @property
    def input_ports(self) -> tuple[str, ...]:
        signals = tuple(dict.fromkeys(signal for source in self.sources for signal in source.signals))
        return (*signals, "temperature") if self.temperature_input else signals

    def derivatives(self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: ReactiveCSTRParams) -> jax.Array:
        feed = inlets["in"]
        temperature = inputs["temperature"] if self.temperature_input else jnp.asarray(params.temperature)
        temperature_k = temperature + ZERO_CELSIUS

        sources = jnp.zeros_like(state)
        for source in self.sources:
            sources = source.add_to(sources, state, {signal: inputs[signal] for signal in source.signals}, temperature_k)
        return feed.flow / params.volume * (feed.components - state) + sources

    def outlet_flows(
        self, state: jax.Array, inlet_flows: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array], params: ReactiveCSTRParams
    ) -> Mapping[str, jax.Array]:
        return {"out": inlet_flows["in"]}

    def outlet_compositions(
        self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: ReactiveCSTRParams
    ) -> Mapping[str, jax.Array]:
        return {"out": state}

    def project(self, state: jax.Array, params: ReactiveCSTRParams) -> jax.Array:
        return self.projection(state)
