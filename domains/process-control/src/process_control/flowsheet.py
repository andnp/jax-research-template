from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.event_registry import compile_registry
from process_control.events import EventPoint
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
from process_control.integration import DEFAULT_BUDGET, NONFINITE_STATE, PROJECTION, AdvanceBudget, GuardedSystem, StageResult, StateCheck, state_check
from process_control.integration import advance as advance_system
from process_control.integration_policy import IntegrationPolicy, legacy_policy
from process_control.integration_records import AdvanceResult, IntegrationContext, check_result, nonfinite_check
from process_control.ledger import LedgerSpec
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


def _select[T](condition: jax.Array, on_true: T, on_false: T) -> T:
    def pick(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.where(condition, a, b)

    return jax.tree.map(pick, on_true, on_false)


def _total(flows: Iterable[jax.Array]) -> jax.Array:
    return sum(flows, jnp.zeros(()))


@dataclass(frozen=True)
class Node[S, P]:
    """A unit placed in a flowsheet, with its parameters and the plant signals routed to its input ports.

    Input ports missing from `inputs` read the plant signal of the same name. `policy` declares the node's
    numerical domain; the default declares none.
    """

    unit: ContinuousUnit[S, P]
    params: P
    inputs: Mapping[str, str] = field(default_factory=dict)
    policy: IntegrationPolicy[S, P] = field(default_factory=legacy_policy)


@jax_dataclass
class FlowsheetReport:
    """Every resolved stream, the external outlets among them, and each group diagnostic keyed by (node, name)."""

    outlets: dict[str, Stream]
    streams: dict[Port, Stream]
    split_deficits: dict[str, jax.Array]
    diagnostics: dict[tuple[str, str], jax.Array]


@jax_dataclass
class RhsEvaluation:
    """One transient stage evaluation: the plant derivative with the streams and diagnostics resolved for it."""

    derivative: Mapping[str, object]
    streams: dict[Port, Stream]
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
    ledgers: tuple[LedgerSpec[Mapping[str, object], "RhsEvaluation"], ...] = ()

    @property
    def stateful_nodes(self) -> tuple[str, ...]:
        return tuple(name for name, node in self.nodes.items() if node.unit.stateful)

    def evaluate_rhs(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> RhsEvaluation:
        """Resolve once at `state` and return the derivative with the streams and diagnostics it was computed from.

        A group failing a declared validity requirement hands downstream groups its declared placeholder
        outputs instead of its own, so invalid constitutive results never feed further physics.
        """
        streams, diagnostics = self._resolve(state, inlets, inputs, gated=True)
        derivative = {
            name: self.nodes[name].unit.derivatives(
                state[name], self._inlets_of(name, streams, inlets), self._inputs_of(name, inputs), self.nodes[name].params
            )
            for name in self.stateful_nodes
        }
        return RhsEvaluation(derivative, streams, diagnostics)

    def derivatives(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> Mapping[str, object]:
        return self.evaluate_rhs(state, inlets, inputs).derivative

    @property
    def mode_ports(self) -> dict[str, Port]:
        """The input port carrying each persistent mode, keyed `node.mode` in declaration order."""
        return {f"{name}.{mode}": Port(name, port) for name, node in self.nodes.items() for mode, port in node.policy.modes.items()}

    def initial_context(self, modes: Mapping[str, bool | int]) -> IntegrationContext:
        """Build the persistent context from explicit initial values for exactly the declared modes."""
        if set(modes) != set(self.mode_ports):
            raise ValueError(f"initial modes must be exactly {sorted(self.mode_ports)}, got {sorted(modes)}")
        return IntegrationContext({name: jnp.asarray(modes[name], dtype=jnp.int32) for name in self.mode_ports})

    def project(self, state: Mapping[str, object]) -> Mapping[str, object]:
        return {name: self.nodes[name].unit.project(state[name], self.nodes[name].params) for name in self.stateful_nodes}

    @property
    def check_names(self) -> tuple[str, ...]:
        """Every guard flag in reason-id order (reason i names entry i - 1): state checks first, then post-evaluation checks."""
        pre = (NONFINITE_STATE, *(f"{name}.{check}" for name, node in self.nodes.items() for check in node.policy.check_names))
        finite = tuple(f"{name}.finite_rhs" for name in self.stateful_nodes if self.nodes[name].policy.finite_rhs)
        validity = tuple(f"{name}.{req.group}.{req.diagnostic}" for name, node in self.nodes.items() for req in node.policy.validity)
        return (*pre, *finite, *validity)

    def advance(
        self,
        state: Mapping[str, object],
        inlets: Mapping[str, Stream],
        inputs: Signals,
        dt: jax.Array | float,
        substeps: int,
        context: IntegrationContext,
        budget: AdvanceBudget = DEFAULT_BUDGET,
    ) -> AdvanceResult[Mapping[str, object]]:
        """Advance the whole plant over dt with the guarded integrator, holding external inlets and input signals constant.

        `context` carries the persistent modes, which reach their units' input ports and stay frozen within
        each smooth segment; pass the returned context to the next call. `report` and `derivatives` read
        the same modes when given `inputs | context.modes`.
        """
        if set(context.modes) != set(self.mode_ports):
            raise ValueError(f"context must hold exactly the modes {sorted(self.mode_ports)}, got {sorted(context.modes)}")
        clash = sorted(set(inputs) & set(self.mode_ports))
        if clash:
            raise ValueError(f"signals {clash} collide with persistent modes")
        return advance_system(self._guarded_system(inlets, inputs), state, context, dt, substeps, budget)

    def report(self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals) -> FlowsheetReport:
        streams, diagnostics = self._resolve(state, inlets, inputs)
        return FlowsheetReport(
            outlets={name: streams[port] for name, port in self.external_outlets.items()},
            streams=streams,
            split_deficits={node: value for (node, name), value in diagnostics.items() if name == SPLIT_DEFICIT},
            diagnostics=diagnostics,
        )

    def _resolve(
        self, state: Mapping[str, object], inlets: Mapping[str, Stream], inputs: Signals, gated: bool = False
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
            outputs = result.values
            for requirement in node.policy.validity if gated else ():
                if requirement.group == group.name and requirement.placeholder is not None:
                    valid = jnp.all(result.diagnostics[group.diagnostic_names.index(requirement.diagnostic)] != 0)
                    outputs = tuple(jnp.where(valid, value, safe) for value, safe in zip(outputs, requirement.placeholder(outputs), strict=True))
            values[gid.node].update(zip(group.outputs, outputs, strict=True))
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

    def _guarded_system(self, inlets: Mapping[str, Stream], inputs: Signals) -> GuardedSystem[Mapping[str, object]]:
        """The plant as the integrator sees it: node checks in fixed order, witnesses for failing nodes, gated stages, ledgers and events."""
        registry = compile_registry({name: node.policy for name, node in self.nodes.items()})
        pre_mask = (False, *(check in node.policy.continuation for node in self.nodes.values() for check in node.policy.check_names))
        stage_mask = pre_mask + (False,) * (len(self.check_names) - len(pre_mask))
        stage_reads = {name: tuple(read for event in node.policy.events for read in event.stage_reads) for name, node in self.nodes.items()}

        def signals(context: IntegrationContext) -> dict[str, jax.Array]:
            return {**inputs, **context.modes}

        def check(state: Mapping[str, object], context: IntegrationContext) -> StateCheck:
            flags, _ = self._state_flags(state, signals(context), context)
            return state_check(check_result((nonfinite_check({NONFINITE_STATE: state}).failed[0], *flags)), pre_mask)

        def stage(state: Mapping[str, object], context: IntegrationContext) -> StageResult[Mapping[str, object]]:
            # Whole-state finiteness is left to the trial endpoint, which any nonfinite stage poisons.
            current = signals(context)
            flags, failed = self._state_flags(state, current, context)
            flags = (jnp.zeros((), bool), *flags)
            safe = self._sanitize(state, failed)
            evaluation = self.evaluate_rhs(safe, inlets, current)
            rates = {ledger.name: ledger.evaluate(safe, evaluation, current, context) for ledger in self.ledgers}
            return StageResult(evaluation.derivative, rates, state_check(check_result((*flags, *self._evaluation_flags(evaluation))), stage_mask))

        def surfaces(state: Mapping[str, object], context: IntegrationContext) -> jax.Array:
            current = signals(context)
            safe = self._sanitize(state, self._state_flags(state, current, context)[1])
            fields: dict[str, dict[LocalField, jax.Array]] = {name: {} for name in self.nodes}
            if any(stage_reads.values()):
                streams, _ = self._resolve(safe, inlets, current, gated=True)
                for name, reads in stage_reads.items():
                    resolved = self._inlets_of(name, streams, inlets) if reads else {}
                    fields[name] = {read: resolved[read.port].value(read.field) for read in reads}
            point = EventPoint(
                state={name: safe.get(name) for name in self.nodes},
                params={name: node.params for name, node in self.nodes.items()},
                context=context,
                inputs={name: self._inputs_of(name, current) for name in self.nodes},
                fields=fields,
            )
            return registry.surfaces(point)

        return GuardedSystem(stage, check, self.project, registry, surfaces if registry.events else None)

    def _state_flags(
        self, state: Mapping[str, object], signals: Signals, context: IntegrationContext
    ) -> tuple[tuple[jax.Array, ...], dict[str, jax.Array]]:
        """Every node's declared checks in node order, and which nodes fail a hard check."""
        flags: list[jax.Array] = []
        failed: dict[str, jax.Array] = {}
        for name, node in self.nodes.items():
            policy = node.policy
            if not policy.check_names:
                continue
            modes = {mode: context.modes[f"{name}.{mode}"] for mode in policy.modes}
            result = policy.check(state.get(name), self._inputs_of(name, signals), node.params, modes)
            hard = jnp.asarray([check not in policy.continuation for check in policy.check_names], dtype=bool)
            flags.extend(result.failed)
            failed[name] = jnp.any(result.failed & hard)
        return tuple(flags), failed

    def _sanitize(self, state: Mapping[str, object], failed: Mapping[str, jax.Array]) -> dict[str, object]:
        """Swap each hard-failing node's state for its policy's witness, so masked lanes never evaluate outside the domain."""
        safe = dict(state)
        for name, flag in failed.items():
            witness = self.nodes[name].policy.witness
            if witness is None or name not in safe:
                continue
            safe[name] = _select(flag, witness(safe[name], self.nodes[name].params), safe[name])
        return safe

    def _evaluation_flags(self, evaluation: RhsEvaluation) -> tuple[jax.Array, ...]:
        finite = (
            nonfinite_check({name: evaluation.derivative[name]}).failed[0] for name in self.stateful_nodes if self.nodes[name].policy.finite_rhs
        )
        validity = (
            ~jnp.all(evaluation.diagnostics[(name, req.diagnostic)] != 0) for name, node in self.nodes.items() for req in node.policy.validity
        )
        return (*finite, *validity)


UNIT = "unit"


def advance_unit[S, P](
    unit: ContinuousUnit[S, P],
    state: S,
    inlets: Mapping[str, Stream],
    inputs: Signals,
    params: P,
    dt: jax.Array | float,
    substeps: int,
    policy: IntegrationPolicy[S, P] | None = None,
    budget: AdvanceBudget = DEFAULT_BUDGET,
) -> AdvanceResult[S]:
    """Advance one unit on its own with the guarded integrator, holding its inlets and inputs constant over dt.

    The unit runs as a one-node plant whose inlets and outlets are all external, so it is the same
    algorithm as a flowsheet advance. A policy declaring modes needs a flowsheet to carry them.
    """
    node = Node(unit, params, policy=policy if policy is not None else legacy_policy())
    plant = build_flowsheet(
        {UNIT: node}, [], {port: Port(UNIT, port) for port in unit.inlet_ports}, {port: Port(UNIT, port) for port in unit.outlet_ports}
    )
    result = plant.advance({UNIT: state}, inlets, inputs, dt, substeps, plant.initial_context({}), budget)
    # The plant state erases node types; its only node holds this unit's S.
    unit_state = cast(S, result.state[UNIT])
    return AdvanceResult(unit_state, result.completed_time_h, result.status, result.quadratures, result.context, result.checks)


def build_flowsheet(
    nodes: Mapping[str, Node],
    connections: Sequence[tuple[Port, Port]],
    inlets: Mapping[str, Port],
    outlets: Mapping[str, Port],
    ledgers: Sequence[LedgerSpec[Mapping[str, object], RhsEvaluation]] = (),
) -> Flowsheet:
    """Validate a plant graph and compile the dependency order of its evaluation groups.

    `connections` are (outlet, inlet) pairs; `inlets` and `outlets` name the external streams and the
    node ports they feed or leave from. Every node inlet must be fed once and every node outlet consumed
    once, so the graph conserves flow. Connected ports must agree on schema and phase. `ledgers` declare
    the plant-level rates whose accepted quadratures an advance returns, keyed by ledger name.
    """
    names = [ledger.name for ledger in ledgers]
    if len(set(names)) != len(names) or PROJECTION in names:
        raise ValueError(f"ledger names must be unique and not {PROJECTION!r}: {names}")
    routes = {name: _route(name, node) for name, node in nodes.items()}
    for name, node in nodes.items():
        _check_groups(name, node)
        _check_policy(name, node)

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
        ledgers=tuple(ledgers),
    )


def _route(name: str, node: Node) -> dict[str, str]:
    """Each input port's plant signal; a mode-owned port reads its `node.mode` context entry instead."""
    unknown = set(node.inputs) - set(node.unit.input_ports)
    if unknown:
        raise ValueError(f"node {name!r} has no input ports {sorted(unknown)}; it has {list(node.unit.input_ports)}")
    modes = {port: f"{name}.{mode}" for mode, port in node.policy.modes.items()}
    return {port: modes.get(port, node.inputs.get(port, port)) for port in node.unit.input_ports}


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


def _check_policy(name: str, node: Node) -> None:
    unit, policy = node.unit, node.policy
    groups = {group.name: group for group in unit.evaluation_groups}
    for requirement in policy.validity:
        if requirement.group not in groups:
            raise ValueError(f"node {name!r}: validity requirement names unknown group {requirement.group!r}")
        if requirement.diagnostic not in groups[requirement.group].diagnostic_names:
            raise ValueError(f"node {name!r}: group {requirement.group!r} declares no diagnostic {requirement.diagnostic!r}")
    for mode, port in policy.modes.items():
        if port not in unit.input_ports:
            raise ValueError(f"node {name!r}: mode {mode!r} names input port {port!r}; the unit has {list(unit.input_ports)}")
        if port in node.inputs:
            raise ValueError(f"node {name!r}: mode {mode!r} owns input port {port!r}, which is also routed to signal {node.inputs[port]!r}")
    for event in policy.events:
        foreign = [str(read) for read in event.stage_reads if read.port not in unit.inlet_ports]
        if foreign:
            raise ValueError(f"node {name!r}: event {event.name!r} reads {foreign}, which are not inlet fields")


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
