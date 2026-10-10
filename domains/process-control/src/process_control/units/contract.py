from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import jax

from process_control.components import ComponentSchema
from process_control.integration import rk4_substeps
from process_control.streams import Stream


@dataclass(frozen=True)
class Outlet:
    """Static declaration of a unit outlet: its schema and the inlet ports its flow and composition read.

    Outlets may be evaluated with placeholder values on every inlet outside these sets, so an outlet
    must not read any inlet it does not declare. Empty sets mean state and inputs only.
    """

    schema: ComponentSchema
    flow_from: frozenset[str] = frozenset()
    composition_from: frozenset[str] = frozenset()


class ContinuousUnit[S, P](Protocol):
    """A continuous-time unit: state derivatives plus outlets resolved in a flow pass then a composition pass."""

    @property
    def inlet_ports(self) -> Mapping[str, ComponentSchema]: ...

    @property
    def outlet_ports(self) -> Mapping[str, Outlet]: ...

    @property
    def input_ports(self) -> tuple[str, ...]: ...

    def derivatives(self, state: S, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: P) -> S: ...

    def outlet_flows(
        self, state: S, inlet_flows: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array], params: P
    ) -> Mapping[str, jax.Array]: ...

    def outlet_compositions(
        self, state: S, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: P
    ) -> Mapping[str, jax.Array]: ...

    def project(self, state: S, params: P) -> S: ...


def unit_outlets[S, P](
    unit: ContinuousUnit[S, P],
    state: S,
    inlets: Mapping[str, Stream],
    inputs: Mapping[str, jax.Array],
    params: P,
) -> dict[str, Stream]:
    """Evaluate every outlet of one unit as a stream, given all of its inlets."""
    inlet_flows = {name: stream.flow for name, stream in inlets.items()}
    flows = unit.outlet_flows(state, inlet_flows, inputs, params)
    compositions = unit.outlet_compositions(state, inlets, inputs, params)
    return {
        name: Stream(flow=flows[name], components=compositions[name], schema=outlet.schema)
        for name, outlet in unit.outlet_ports.items()
    }


def step_unit[S, P](
    unit: ContinuousUnit[S, P],
    state: S,
    inlets: Mapping[str, Stream],
    inputs: Mapping[str, jax.Array],
    params: P,
    dt: jax.Array,
    substeps: int,
) -> S:
    """Integrate one unit on its own, holding its inlets and inputs constant over dt."""

    def derivatives(current: S) -> S:
        return unit.derivatives(current, inlets, inputs, params)

    def project(current: S) -> S:
        return unit.project(current, params)

    return rk4_substeps(derivatives, project, state, dt, substeps)
