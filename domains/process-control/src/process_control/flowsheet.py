from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.field_graph import CompiledGroups, External, GroupId, Source, compile_groups
from process_control.fields import (
    CONTEXT_TEMPERATURE,
    EvaluationGroup,
    FieldRef,
    GroupResult,
    LocalField,
    PortSpec,
    PrescribedContext,
    StreamField,
    validate_groups,
)
from process_control.integration import rk4_substeps
from process_control.streams import Stream, mix_components
from process_control.units.contract import ContinuousUnit, evaluate, outlet_streams

type Signals = Mapping[str, jax.Array]


class Port(NamedTuple):
    node: str
    port: str

    def __str__(self) -> str:
        return f"{self.node}.{self.port}"


FLOW = StreamField.FLOW
COMPONENTS = StreamField.COMPONENTS
SPLIT_DEFICIT = "split_deficit"


@dataclass(frozen=True)
class Mixer:
    """Stateless junction combining same-phase inlets into one outlet at a prescribed temperature and pressure."""

    port: PortSpec
    inlets: tuple[str, ...]
    context: PrescribedContext

    @property
    def inlet_ports(self) -> Mapping[str, PortSpec]:
        return dict.fromkeys(self.inlets, self.port)

    @property
    def outlet_ports(self) -> Mapping[str, PortSpec]:
        return {"out": self.port}

    @property
    def evaluation_groups(self) -> tuple[EvaluationGroup, ...]:
        flows = tuple(LocalField(name, FLOW) for name in self.inlets)
        compositions = tuple(LocalField(name, COMPONENTS) for name in self.inlets)
        return (
            EvaluationGroup("flow", outputs=(LocalField("out", FLOW),), reads=flows),
            EvaluationGroup("composition", outputs=(LocalField("out", COMPONENTS),), reads=flows + compositions),
            self.context.group(("out",)),
        )

    @property
    def input_ports(self) -> tuple[str, ...]:
        return (CONTEXT_TEMPERATURE,)

    @property
    def stateful(self) -> bool:
        return False

    def evaluate_group(self, group_name: str, state: None, reads: Mapping[LocalField, jax.Array], inputs: Signals, params: None) -> GroupResult:
        match group_name:
            case "flow":
                return GroupResult((jnp.sum(jnp.stack([reads[LocalField(name, FLOW)] for name in self.inlets])),))
            case "composition":
                flows = [reads[LocalField(name, FLOW)] for name in self.inlets]
                compositions = [reads[LocalField(name, COMPONENTS)] for name in self.inlets]
                return GroupResult((mix_components(self.port.phase, flows, compositions),))
            case "context":
                return self.context.evaluate(("out",), inputs)
        raise ValueError(f"mixer has no group {group_name!r}")

    def derivatives(self, state: None, inlets: Mapping[str, Stream], inputs: Signals, params: None) -> None:
        return None

    def project(self, state: None, params: None) -> None:
        return None


@dataclass(frozen=True)
class Splitter:
    """Stateless junction drawing input-set flows from its inlet; the remainder outlet takes the rest.

    Draws are clamped at zero but not capped by the inlet flow, because a cap inside a recycle is a
    genuine algebraic loop. The remainder is clamped at zero instead, and whatever the split could not
    honour is reported as its deficit. Every outlet carries the inlet's composition, temperature and pressure.
    """

    port: PortSpec
    draws: tuple[str, ...]
    remainder: str

    def __post_init__(self) -> None:
        names = (*self.draws, self.remainder)
        if len(set(names)) != len(names):
            raise ValueError(f"splitter outlet names must be unique: {names}")

    @property
    def inlet_ports(self) -> Mapping[str, PortSpec]:
        return {"in": self.port}

    @property
    def outlet_ports(self) -> Mapping[str, PortSpec]:
        return dict.fromkeys(self._outlets, self.port)

    @property
    def evaluation_groups(self) -> tuple[EvaluationGroup, ...]:
        def forwarded(name: str, kind: StreamField) -> EvaluationGroup:
            return EvaluationGroup(name, outputs=tuple(LocalField(port, kind) for port in self._outlets), reads=(LocalField("in", kind),))

        return (
            EvaluationGroup("draw_flows", outputs=tuple(LocalField(name, FLOW) for name in self.draws), reads=(), input_names=self.draws),
            EvaluationGroup(
                "remainder_flow",
                outputs=(LocalField(self.remainder, FLOW),),
                reads=(LocalField("in", FLOW),),
                input_names=self.draws,
                diagnostic_names=(SPLIT_DEFICIT,),
            ),
            forwarded("composition", COMPONENTS),
            forwarded("temperature", StreamField.TEMPERATURE),
            forwarded("pressure", StreamField.PRESSURE),
        )

    @property
    def input_ports(self) -> tuple[str, ...]:
        return self.draws

    @property
    def stateful(self) -> bool:
        return False

    def evaluate_group(self, group_name: str, state: None, reads: Mapping[LocalField, jax.Array], inputs: Signals, params: None) -> GroupResult:
        match group_name:
            case "draw_flows":
                return GroupResult(tuple(self._draws(inputs).values()))
            case "remainder_flow":
                draws = self._draws(inputs)
                inflow = reads[LocalField("in", FLOW)]
                refused = _total(draws[name] - inputs[name] for name in self.draws)
                deficit = refused + jnp.maximum(_total(draws.values()) - inflow, 0.0)
                return GroupResult((jnp.maximum(inflow - _total(draws.values()), 0.0),), (deficit,))
            case "composition":
                return GroupResult((reads[LocalField("in", COMPONENTS)],) * len(self._outlets))
            case "temperature":
                return GroupResult((reads[LocalField("in", StreamField.TEMPERATURE)],) * len(self._outlets))
            case "pressure":
                return GroupResult((reads[LocalField("in", StreamField.PRESSURE)],) * len(self._outlets))
        raise ValueError(f"splitter has no group {group_name!r}")

    def derivatives(self, state: None, inlets: Mapping[str, Stream], inputs: Signals, params: None) -> None:
        return None

    def project(self, state: None, params: None) -> None:
        return None

    @property
    def _outlets(self) -> tuple[str, ...]:
        return (*self.draws, self.remainder)

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
    """Every resolved stream, the external outlets among them, and each group diagnostic keyed by (node, name)."""

    outlets: dict[str, Stream]
    streams: dict[Port, Stream]
    split_deficits: dict[str, jax.Array]
    diagnostics: dict[tuple[str, str], jax.Array]


@dataclass(frozen=True, eq=False)
class Flowsheet:
    """A validated plant graph integrated as one coupled ODE over a mapping of node states.

    Build it with `build_flowsheet`. The plant state holds one entry per stateful node. Every resolution
    evaluates each compiled evaluation group once, in dependency order, and keeps nothing between calls.
    Hashed by identity so it can be a static argument to `jax.jit`.
    """

    nodes: Mapping[str, Node]
    sources: Mapping[Port, Port | str]
    external_inlets: Mapping[str, Port]
    external_outlets: Mapping[str, Port]
    routes: Mapping[str, Mapping[str, str]]
    groups: Mapping[GroupId, EvaluationGroup]
    plan: CompiledGroups

    @property
    def stateful_nodes(self) -> tuple[str, ...]:
        return tuple(name for name, node in self.nodes.items() if node.unit.stateful)

    def derivatives(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> Mapping[str, object]:
        streams, _ = self._resolve(state, inlets, inputs)
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
        streams, diagnostics = self._resolve(state, inlets, inputs)
        return FlowsheetReport(
            outlets={name: streams[port] for name, port in self.external_outlets.items()},
            streams=streams,
            split_deficits={node: value for (node, name), value in diagnostics.items() if name == SPLIT_DEFICIT},
            diagnostics=diagnostics,
        )

    def _resolve(
        self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals
    ) -> tuple[dict[Port, Stream], dict[tuple[str, str], jax.Array]]:
        if set(state) != set(self.stateful_nodes):
            raise ValueError(f"plant state must hold exactly the nodes {sorted(self.stateful_nodes)}, got {sorted(state)}")
        for name, port in self.external_inlets.items():
            expected = self.nodes[port.node].unit.inlet_ports[port.port]
            if name not in inlets:
                raise ValueError(f"missing external inlet {name!r}")
            if inlets[name].schema != expected.schema:
                raise ValueError(f"external inlet {name!r} carries {inlets[name].schema.names} but {port} expects {expected.schema.names}")
            if inlets[name].phase is not expected.phase:
                raise ValueError(f"external inlet {name!r} is {inlets[name].phase} but {port} expects {expected.phase}")

        values: dict[str, dict[LocalField, jax.Array]] = {name: {} for name in self.nodes}
        diagnostics: dict[tuple[str, str], jax.Array] = {}
        for gid in self.plan.order:
            node, group = self.nodes[gid.node], self.groups[gid]
            reads = {
                read: self._source_value(source, values, inlets)
                for read, source in zip(group.reads, self.plan.sources[gid], strict=True)
            }
            group_inputs = {name: inputs[self.routes[gid.node][name]] for name in group.input_names}
            result = evaluate(node.unit, group, state.get(gid.node), reads, group_inputs, node.params)
            values[gid.node].update(zip(group.outputs, result.values, strict=True))
            diagnostics.update(((gid.node, name), value) for name, value in zip(group.diagnostic_names, result.diagnostics, strict=True))

        streams = {
            Port(name, port): stream
            for name, node in self.nodes.items()
            for port, stream in outlet_streams(node.unit.outlet_ports, values[name]).items()
        }
        return streams, diagnostics

    def _source_value(self, source: Source, values: Mapping[str, Mapping[LocalField, jax.Array]], inlets: Mapping[str, Stream]) -> jax.Array:
        if isinstance(source, External):
            external = self.sources[Port(source.inlet.node, source.inlet.port)]
            assert isinstance(external, str)
            return inlets[external].value(source.inlet.field)
        return values[source.node][LocalField(source.port, source.field)]

    def _inlets_of(self, name: str, streams: Mapping[Port, Stream], inlets: Mapping[str, Stream]) -> dict[str, Stream]:
        resolved: dict[str, Stream] = {}
        for port in self.nodes[name].unit.inlet_ports:
            source = self.sources[Port(name, port)]
            resolved[port] = inlets[source] if isinstance(source, str) else streams[source]
        return resolved

    def _inputs_of(self, name: str, inputs: Signals) -> dict[str, jax.Array]:
        return {port: inputs[signal] for port, signal in self.routes[name].items()}


def build_flowsheet(
    nodes: Mapping[str, Node],
    connections: Sequence[tuple[Port, Port]],
    inlets: Mapping[str, Port],
    outlets: Mapping[str, Port],
) -> Flowsheet:
    """Validate a plant graph and compile the dependency order of its evaluation groups.

    `connections` are (outlet, inlet) pairs; `inlets` and `outlets` name the external streams and the
    node ports they feed or leave from. Every node inlet must be fed once and every node outlet consumed
    once, so the graph conserves flow. Connected ports must agree on schema and phase.
    """
    routes = {name: _route(name, node) for name, node in nodes.items()}
    for name, node in nodes.items():
        _check_groups(name, node)

    sources: dict[Port, Port | str] = {}
    consumers: dict[Port, str] = {}
    for source, target in connections:
        _claim(consumers, _outlet(nodes, source), str(target))
        _claim(sources, _inlet(nodes, target), source)
        _check_compatible(source, nodes[source.node].unit.outlet_ports[source.port], target, nodes[target.node].unit.inlet_ports[target.port])
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

    internal = {(inlet.node, inlet.port): (source.node, source.port) for inlet, source in sources.items() if isinstance(source, Port)}
    external = {FieldRef(port.node, port.port, kind) for port in inlets.values() for kind in StreamField}
    plan = compile_groups({name: node.unit.evaluation_groups for name, node in nodes.items()}, internal, external)
    groups = {GroupId(name, group.name): group for name, node in nodes.items() for group in node.unit.evaluation_groups}

    return Flowsheet(
        nodes=nodes,
        sources=sources,
        external_inlets=inlets,
        external_outlets=outlets,
        routes=routes,
        groups=groups,
        plan=plan,
    )


def _route(name: str, node: Node) -> dict[str, str]:
    unknown = set(node.inputs) - set(node.unit.input_ports)
    if unknown:
        raise ValueError(f"node {name!r} has no input ports {sorted(unknown)}; it has {list(node.unit.input_ports)}")
    return {port: node.inputs.get(port, port) for port in node.unit.input_ports}


def _check_groups(name: str, node: Node) -> None:
    unit = node.unit
    try:
        validate_groups(unit.evaluation_groups, unit.inlet_ports, unit.outlet_ports)
    except ValueError as error:
        raise ValueError(f"node {name!r}: {error}") from error
    for group in unit.evaluation_groups:
        unknown = sorted(set(group.input_names) - set(unit.input_ports))
        if unknown:
            raise ValueError(f"node {name!r}: group {group.name!r} reads undeclared input ports {unknown}")


def _check_compatible(source: Port, carried: PortSpec, target: Port, expected: PortSpec) -> None:
    if carried.phase is not expected.phase:
        raise ValueError(f"phase mismatch: {source} carries {carried.phase} but {target} expects {expected.phase}")
    if carried.schema != expected.schema:
        raise ValueError(f"schema mismatch: {source} carries {carried.schema.names} but {target} expects {expected.schema.names}")


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
