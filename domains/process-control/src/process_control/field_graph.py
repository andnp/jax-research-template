import heapq
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import NamedTuple

from process_control.fields import EvaluationGroup, FieldRef


class GroupId(NamedTuple):
    node: str
    group: str

    def __str__(self) -> str:
        return f"{self.node}.{self.group}"


@dataclass(frozen=True)
class External:
    """A read satisfied by a field supplied from outside the flowsheet."""

    inlet: FieldRef


type Source = FieldRef | External


@dataclass(frozen=True)
class CompiledGroups:
    """Groups in dependency order, the group owning each outlet field, and where each group's reads come from.

    `sources[group]` is aligned with that group's declared `reads`.
    """

    order: tuple[GroupId, ...]
    owners: Mapping[FieldRef, GroupId]
    sources: Mapping[GroupId, tuple[Source, ...]]


def compile_groups(
    groups: Mapping[str, Sequence[EvaluationGroup]],
    connections: Mapping[tuple[str, str], tuple[str, str]],
    external: Set[FieldRef] = frozenset(),
) -> CompiledGroups:
    """Order every node's evaluation groups so each runs after the groups owning the fields it reads.

    `connections` maps a consuming (node, inlet port) to its producing (node, outlet port); inlet fields of
    unconnected ports must appear in `external`. The order follows declaration order wherever dependencies
    allow. A dependency cycle raises ValueError listing its groups and field edges.
    """
    declared = {GroupId(node, group.name): group for node, node_groups in groups.items() for group in node_groups}
    ids = list(declared)
    owners = _owners(declared)
    sources = {gid: _resolve_reads(gid, group, connections, external, owners) for gid, group in declared.items()}

    producers = {
        gid: tuple(owners[source] for source in group_sources if isinstance(source, FieldRef))
        for gid, group_sources in sources.items()
    }
    order = _schedule(ids, producers)
    if len(order) < len(ids):
        raise ValueError(_cycle_message(ids, order, sources, owners))
    return CompiledGroups(order=tuple(order), owners=owners, sources=sources)


def _owners(declared: Mapping[GroupId, EvaluationGroup]) -> dict[FieldRef, GroupId]:
    owners: dict[FieldRef, GroupId] = {}
    for gid, group in declared.items():
        for output in group.outputs:
            ref = FieldRef(gid.node, output.port, output.field)
            if ref in owners:
                raise ValueError(f"{ref} is owned by both {owners[ref]} and {gid}")
            owners[ref] = gid
    return owners


def _resolve_reads(
    gid: GroupId,
    group: EvaluationGroup,
    connections: Mapping[tuple[str, str], tuple[str, str]],
    external: Set[FieldRef],
    owners: Mapping[FieldRef, GroupId],
) -> tuple[Source, ...]:
    resolved: list[Source] = []
    for read in group.reads:
        inlet = FieldRef(gid.node, read.port, read.field)
        producer = connections.get((gid.node, read.port))
        if producer is None:
            if inlet not in external:
                raise ValueError(f"{gid} reads {inlet}, which is neither connected nor external")
            resolved.append(External(inlet))
            continue

        source = FieldRef(*producer, read.field)
        if source not in owners:
            raise ValueError(f"{gid} reads {inlet} from {source}, which no group owns")
        resolved.append(source)
    return tuple(resolved)


def _schedule(ids: Sequence[GroupId], producers: Mapping[GroupId, tuple[GroupId, ...]]) -> list[GroupId]:
    """Kahn's algorithm, always releasing the earliest-declared ready group."""
    position = {gid: index for index, gid in enumerate(ids)}
    waiting = {gid: len(set(producers[gid])) for gid in ids}
    consumers: dict[GroupId, list[GroupId]] = {gid: [] for gid in ids}
    for gid in ids:
        for producer in dict.fromkeys(producers[gid]):
            consumers[producer].append(gid)

    ready = [position[gid] for gid in ids if waiting[gid] == 0]
    heapq.heapify(ready)
    order: list[GroupId] = []
    while ready:
        gid = ids[heapq.heappop(ready)]
        order.append(gid)
        for consumer in consumers[gid]:
            waiting[consumer] -= 1
            if waiting[consumer] == 0:
                heapq.heappush(ready, position[consumer])
    return order


def _cycle_message(
    ids: Sequence[GroupId],
    order: Sequence[GroupId],
    sources: Mapping[GroupId, tuple[Source, ...]],
    owners: Mapping[FieldRef, GroupId],
) -> str:
    scheduled = set(order)
    stuck = [gid for gid in ids if gid not in scheduled]

    def next_edge(gid: GroupId) -> tuple[FieldRef, GroupId]:
        for source in sources[gid]:
            if isinstance(source, FieldRef) and owners[source] not in scheduled:
                return source, owners[source]
        raise AssertionError(f"{gid} is unscheduled but has no unscheduled producer")

    walk = [stuck[0]]
    edges: list[tuple[GroupId, FieldRef]] = []
    while True:
        source, producer = next_edge(walk[-1])
        edges.append((walk[-1], source))
        if producer in walk:
            start = walk.index(producer)
            cycle = edges[start:]
            break
        walk.append(producer)

    steps = " -> ".join(f"{consumer} reads {source} (owned by {owners[source]})" for consumer, source in cycle)
    return f"flowsheet field dependencies form a cycle: {steps}"
