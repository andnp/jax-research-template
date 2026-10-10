from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.components import ComponentSchema
from process_control.integration import rk4_substeps
from process_control.streams import Stream, mix
from process_control.units.contract import ContinuousUnit, Outlet

type Signals = Mapping[str, jax.Array]


class Port(NamedTuple):
    node: str
    port: str

    def __str__(self) -> str:
        return f"{self.node}.{self.port}"


@dataclass(frozen=True)
class Mixer:
    """Stateless junction combining its inlets into one flow-weighted outlet."""

    schema: ComponentSchema
    inlets: tuple[str, ...]

    @property
    def inlet_ports(self) -> Mapping[str, ComponentSchema]:
        return dict.fromkeys(self.inlets, self.schema)

    @property
    def outlet_ports(self) -> Mapping[str, Outlet]:
        return {"out": Outlet(self.schema, flow_from=frozenset(self.inlets), composition_from=frozenset(self.inlets))}

    @property
    def input_ports(self) -> tuple[str, ...]:
        return ()

    def derivatives(self, state: None, inlets: Mapping[str, Stream], inputs: Signals, params: None) -> None:
        return None

    def outlet_flows(self, state: None, inlet_flows: Mapping[str, jax.Array], inputs: Signals, params: None) -> Mapping[str, jax.Array]:
        return {"out": jnp.sum(jnp.stack([inlet_flows[name] for name in self.inlets]))}

    def outlet_compositions(self, state: None, inlets: Mapping[str, Stream], inputs: Signals, params: None) -> Mapping[str, jax.Array]:
        return {"out": mix(*(inlets[name] for name in self.inlets)).components}

    def project(self, state: None, params: None) -> None:
        return None


@dataclass(frozen=True)
class Splitter:
    """Stateless junction drawing input-set flows from its inlet; the remainder outlet takes the rest.

    Draws are clamped at zero but not capped by the inlet flow, because a cap inside a recycle is a
    genuine algebraic loop. The remainder is clamped at zero instead, and whatever the split could not
    honour is reported as its deficit.
    """

    schema: ComponentSchema
    draws: tuple[str, ...]
    remainder: str

    def __post_init__(self) -> None:
        names = (*self.draws, self.remainder)
        if len(set(names)) != len(names):
            raise ValueError(f"splitter outlet names must be unique: {names}")

    @property
    def inlet_ports(self) -> Mapping[str, ComponentSchema]:
        return {"in": self.schema}

    @property
    def outlet_ports(self) -> Mapping[str, Outlet]:
        feed = frozenset({"in"})
        draws = {name: Outlet(self.schema, composition_from=feed) for name in self.draws}
        return {**draws, self.remainder: Outlet(self.schema, flow_from=feed, composition_from=feed)}

    @property
    def input_ports(self) -> tuple[str, ...]:
        return self.draws

    def derivatives(self, state: None, inlets: Mapping[str, Stream], inputs: Signals, params: None) -> None:
        return None

    def outlet_flows(self, state: None, inlet_flows: Mapping[str, jax.Array], inputs: Signals, params: None) -> Mapping[str, jax.Array]:
        draws = self._draws(inputs)
        return {**draws, self.remainder: jnp.maximum(inlet_flows["in"] - _total(draws.values()), 0.0)}

    def outlet_compositions(self, state: None, inlets: Mapping[str, Stream], inputs: Signals, params: None) -> Mapping[str, jax.Array]:
        return {name: inlets["in"].components for name in (*self.draws, self.remainder)}

    def project(self, state: None, params: None) -> None:
        return None

    def deficit(self, inlet_flow: jax.Array, inputs: Signals) -> jax.Array:
        """Requested flow the split cannot honour: negative draws plus draws beyond the inflow."""
        draws = self._draws(inputs)
        refused = _total(draws[name] - inputs[name] for name in self.draws)
        return refused + jnp.maximum(_total(draws.values()) - inlet_flow, 0.0)

    def _draws(self, inputs: Signals) -> dict[str, jax.Array]:
        return {name: jnp.maximum(inputs[name], 0.0) for name in self.draws}


def _total(flows: Iterable[jax.Array]) -> jax.Array:
    return sum(flows, jnp.zeros(()))


@dataclass(frozen=True)
class Node[S, P]:
    """A unit placed in a flowsheet, with its parameters and the plant signals routed to its input ports.

    Input ports missing from `inputs` read the plant signal of the same name.
    """

    unit: ContinuousUnit[S, P]
    params: P
    inputs: Mapping[str, str] = field(default_factory=dict)


@jax_dataclass
class FlowsheetReport:
    outlets: dict[str, Stream]
    streams: dict[Port, Stream]
    split_deficits: dict[str, jax.Array]


@dataclass(frozen=True, eq=False)
class Flowsheet:
    """A validated plant graph integrated as one coupled ODE over a mapping of node states.

    Build it with `build_flowsheet`. The plant state holds one entry per node that is not a junction.
    Hashed by identity so it can be a static argument to `jax.jit`.
    """

    nodes: Mapping[str, Node]
    sources: Mapping[Port, Port | str]
    external_inlets: Mapping[str, Port]
    external_outlets: Mapping[str, Port]
    routes: Mapping[str, Mapping[str, str]]
    flow_order: tuple[Port, ...]
    composition_order: tuple[Port, ...]

    @property
    def stateful_nodes(self) -> tuple[str, ...]:
        return tuple(name for name, node in self.nodes.items() if not isinstance(node.unit, Mixer | Splitter))

    def derivatives(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> Mapping[str, object]:
        streams = self._resolve(state, inlets, inputs)
        return {
            name: self.nodes[name].unit.derivatives(
                state[name], self._inlets_of(name, streams, inlets), self._inputs_of(name, inputs), self.nodes[name].params
            )
            for name in self.stateful_nodes
        }

    def project(self, state: Mapping[str, object]) -> Mapping[str, object]:
        return {name: self.nodes[name].unit.project(state[name], self.nodes[name].params) for name in self.stateful_nodes}

    def step(
        self,
        state: Mapping[str, object],
        inlets: Mapping[str, Stream],
        inputs: Signals,
        dt: jax.Array,
        substeps: int,
    ) -> Mapping[str, object]:
        """Integrate the whole plant over dt, holding external inlets and input signals constant."""
        return rk4_substeps(self.derivatives, self.project, state, dt, substeps, inlets, inputs)

    def report(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> FlowsheetReport:
        streams = self._resolve(state, inlets, inputs)
        deficits = {
            name: node.unit.deficit(self._inlets_of(name, streams, inlets)["in"].flow, self._inputs_of(name, inputs))
            for name, node in self.nodes.items()
            if isinstance(node.unit, Splitter)
        }
        return FlowsheetReport(
            outlets={name: streams[port] for name, port in self.external_outlets.items()},
            streams=streams,
            split_deficits=deficits,
        )

    def _resolve(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> dict[Port, Stream]:
        if set(state) != set(self.stateful_nodes):
            raise ValueError(f"plant state must hold exactly the nodes {sorted(self.stateful_nodes)}, got {sorted(state)}")
        for name, port in self.external_inlets.items():
            expected = self.nodes[port.node].unit.inlet_ports[port.port]
            if name not in inlets:
                raise ValueError(f"missing external inlet {name!r}")
            if inlets[name].schema != expected:
                raise ValueError(f"external inlet {name!r} carries {inlets[name].schema.names} but {port} expects {expected.names}")

        flows: dict[Port, jax.Array] = {}
        flow_calls: dict[tuple[str, frozenset[str]], Mapping[str, jax.Array]] = {}
        for port in self.flow_order:
            node = self.nodes[port.node]
            call = (port.node, node.unit.outlet_ports[port.port].flow_from)
            if call not in flow_calls:
                inlet_flows = {name: self._inlet_flow(Port(port.node, name), flows, inlets) for name in node.unit.inlet_ports}
                flow_calls[call] = node.unit.outlet_flows(state.get(port.node), inlet_flows, self._inputs_of(port.node, inputs), node.params)
            flows[port] = flow_calls[call][port.port]

        streams: dict[Port, Stream] = {}
        composition_calls: dict[tuple[str, frozenset[str]], Mapping[str, jax.Array]] = {}
        for port in self.composition_order:
            node = self.nodes[port.node]
            outlet = node.unit.outlet_ports[port.port]
            call = (port.node, outlet.composition_from)
            if call not in composition_calls:
                node_inlets = {name: self._inlet_stream(Port(port.node, name), flows, streams, inlets) for name in node.unit.inlet_ports}
                composition_calls[call] = node.unit.outlet_compositions(
                    state.get(port.node), node_inlets, self._inputs_of(port.node, inputs), node.params
                )
            streams[port] = Stream(flow=flows[port], components=composition_calls[call][port.port], schema=outlet.schema)
        return streams

    def _inlet_flow(self, inlet: Port, flows: Mapping[Port, jax.Array], inlets: Mapping[str, Stream]) -> jax.Array:
        source = self.sources[inlet]
        if isinstance(source, str):
            return inlets[source].flow
        # Placeholder for an inlet the outlet being evaluated does not declare, so must not read.
        return flows.get(source, jnp.zeros(()))

    def _inlet_stream(
        self, inlet: Port, flows: Mapping[Port, jax.Array], streams: Mapping[Port, Stream], inlets: Mapping[str, Stream]
    ) -> Stream:
        source = self.sources[inlet]
        if isinstance(source, str):
            return inlets[source]
        if source in streams:
            return streams[source]
        schema = self.nodes[source.node].unit.outlet_ports[source.port].schema
        return Stream(flow=flows[source], components=jnp.zeros(len(schema.names)), schema=schema)

    def _inlets_of(self, name: str, streams: Mapping[Port, Stream], inlets: Mapping[str, Stream]) -> dict[str, Stream]:
        return {port: self._inlet_stream(Port(name, port), {}, streams, inlets) for port in self.nodes[name].unit.inlet_ports}

    def _inputs_of(self, name: str, inputs: Signals) -> dict[str, jax.Array]:
        return {port: inputs[signal] for port, signal in self.routes[name].items()}


def build_flowsheet(
    nodes: Mapping[str, Node],
    connections: Sequence[tuple[Port, Port]],
    inlets: Mapping[str, Port],
    outlets: Mapping[str, Port],
) -> Flowsheet:
    """Validate a plant graph and fix its evaluation order.

    `connections` are (outlet, inlet) pairs; `inlets` and `outlets` name the external streams and the
    node ports they feed or leave from. Every node inlet must be fed once and every node outlet consumed
    once, so the graph conserves flow.
    """
    routes = {name: _route(name, node) for name, node in nodes.items()}
    for name, node in nodes.items():
        _check_dependencies(name, node)

    sources: dict[Port, Port | str] = {}
    consumers: dict[Port, str] = {}
    for source, target in connections:
        _claim(consumers, _outlet(nodes, source), str(target))
        _claim(sources, _inlet(nodes, target), source)
        carried = nodes[source.node].unit.outlet_ports[source.port].schema
        expected = nodes[target.node].unit.inlet_ports[target.port]
        if carried != expected:
            raise ValueError(f"schema mismatch: {source} carries {carried.names} but {target} expects {expected.names}")
    for name, target in inlets.items():
        _claim(sources, _inlet(nodes, target), name)
    for name, source in outlets.items():
        _claim(consumers, _outlet(nodes, source), f"external outlet {name!r}")

    unfed = [str(Port(name, port)) for name, node in nodes.items() for port in node.unit.inlet_ports if Port(name, port) not in sources]
    if unfed:
        raise ValueError(f"unconnected inlets: {', '.join(unfed)}")
    unused = [str(Port(name, port)) for name, node in nodes.items() for port in node.unit.outlet_ports if Port(name, port) not in consumers]
    if unused:
        raise ValueError(f"unconnected outlets: {', '.join(unused)}")

    return Flowsheet(
        nodes=nodes,
        sources=sources,
        external_inlets=inlets,
        external_outlets=outlets,
        routes=routes,
        flow_order=_schedule(nodes, sources, flow=True),
        composition_order=_schedule(nodes, sources, flow=False),
    )


def _route(name: str, node: Node) -> dict[str, str]:
    unknown = set(node.inputs) - set(node.unit.input_ports)
    if unknown:
        raise ValueError(f"node {name!r} has no input ports {sorted(unknown)}; it has {list(node.unit.input_ports)}")
    return {port: node.inputs.get(port, port) for port in node.unit.input_ports}


def _check_dependencies(name: str, node: Node) -> None:
    for port, outlet in node.unit.outlet_ports.items():
        unknown = (outlet.flow_from | outlet.composition_from) - set(node.unit.inlet_ports)
        if unknown:
            raise ValueError(f"outlet {name}.{port} depends on unknown inlets {sorted(unknown)}")


def _outlet(nodes: Mapping[str, Node], port: Port) -> Port:
    if port.node not in nodes:
        raise ValueError(f"unknown node {port.node!r} in {port}")
    if port.port not in nodes[port.node].unit.outlet_ports:
        raise ValueError(f"node {port.node!r} has no outlet {port.port!r}; it has {list(nodes[port.node].unit.outlet_ports)}")
    return port


def _inlet(nodes: Mapping[str, Node], port: Port) -> Port:
    if port.node not in nodes:
        raise ValueError(f"unknown node {port.node!r} in {port}")
    if port.port not in nodes[port.node].unit.inlet_ports:
        raise ValueError(f"node {port.node!r} has no inlet {port.port!r}; it has {list(nodes[port.node].unit.inlet_ports)}")
    return port


def _claim[V](claims: dict[Port, V], port: Port, claimant: V) -> None:
    if port in claims:
        raise ValueError(f"{port} is connected twice: to {claims[port]} and {claimant}")
    claims[port] = claimant


def _schedule(nodes: Mapping[str, Node], sources: Mapping[Port, Port | str], *, flow: bool) -> tuple[Port, ...]:
    """Order outlet ports so each follows the outlets it depends on algebraically, rejecting cycles."""

    def dependencies(port: Port) -> list[Port]:
        outlet = nodes[port.node].unit.outlet_ports[port.port]
        reads = outlet.flow_from if flow else outlet.composition_from
        upstream = (sources[Port(port.node, inlet)] for inlet in sorted(reads))
        return [source for source in upstream if isinstance(source, Port)]

    order: list[Port] = []
    done: set[Port] = set()
    path: list[Port] = []

    def visit(port: Port) -> None:
        if port in done:
            return
        if port in path:
            cycle = [*path[path.index(port) :], port][::-1]
            kind = "flow" if flow else "composition"
            raise ValueError(f"algebraic loop in {kind}: {' -> '.join(map(str, cycle))}")
        path.append(port)
        for dependency in dependencies(port):
            visit(dependency)
        path.pop()
        done.add(port)
        order.append(port)

    for name, node in nodes.items():
        for port in node.unit.outlet_ports:
            visit(Port(name, port))
    return tuple(order)
