from collections.abc import Mapping
from typing import Protocol

import jax

from process_control.fields import EvaluationGroup, GroupResult, LocalField, PortSpec, StreamField
from process_control.integration import rk4_substeps
from process_control.streams import Stream


class ContinuousUnit[S, P](Protocol):
    """A continuous-time unit: state derivatives plus outlet fields resolved by atomic evaluation groups.

    Each group owns some outlet fields and reads exactly its declared inlet fields and input signals. A unit
    that is not `stateful` holds no plant state and receives None for it.
    """

    @property
    def inlet_ports(self) -> Mapping[str, PortSpec]: ...

    @property
    def outlet_ports(self) -> Mapping[str, PortSpec]: ...

    @property
    def evaluation_groups(self) -> tuple[EvaluationGroup, ...]: ...

    @property
    def input_ports(self) -> tuple[str, ...]: ...

    @property
    def stateful(self) -> bool: ...

    def evaluate_group(
        self, group_name: str, state: S, reads: Mapping[LocalField, jax.Array], inputs: Mapping[str, jax.Array], params: P
    ) -> GroupResult: ...

    def derivatives(self, state: S, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: P) -> S: ...

    def project(self, state: S, params: P) -> S: ...


def evaluate[S, P](
    unit: ContinuousUnit[S, P],
    group: EvaluationGroup,
    state: S,
    reads: Mapping[LocalField, jax.Array],
    inputs: Mapping[str, jax.Array],
    params: P,
) -> GroupResult:
    """Evaluate one group, checking it returns a value per output and a diagnostic per declared name."""
    result = unit.evaluate_group(group.name, state, reads, inputs, params)
    if len(result.values) != len(group.outputs) or len(result.diagnostics) != len(group.diagnostic_names):
        raise ValueError(
            f"group {group.name!r} returned {len(result.values)} values and {len(result.diagnostics)} diagnostics "
            f"for {len(group.outputs)} outputs and {len(group.diagnostic_names)} diagnostic names"
        )
    return result


def outlet_streams(ports: Mapping[str, PortSpec], values: Mapping[LocalField, jax.Array]) -> dict[str, Stream]:
    """Assemble complete outlet streams from resolved outlet fields."""
    return {
        port: Stream(
            flow=values[LocalField(port, StreamField.FLOW)],
            components=values[LocalField(port, StreamField.COMPONENTS)],
            temperature_k=values[LocalField(port, StreamField.TEMPERATURE)],
            pressure_pa=values[LocalField(port, StreamField.PRESSURE)],
            schema=spec.schema,
            phase=spec.phase,
        )
        for port, spec in ports.items()
    }


def unit_outlets[S, P](
    unit: ContinuousUnit[S, P],
    state: S,
    inlets: Mapping[str, Stream],
    inputs: Mapping[str, jax.Array],
    params: P,
) -> dict[str, Stream]:
    """Evaluate every outlet of one unit as a stream, given all of its inlets."""
    values: dict[LocalField, jax.Array] = {}
    for group in unit.evaluation_groups:
        reads = {read: inlets[read.port].value(read.field) for read in group.reads}
        result = evaluate(unit, group, state, reads, {name: inputs[name] for name in group.input_names}, params)
        values.update(zip(group.outputs, result.values, strict=True))
    return outlet_streams(unit.outlet_ports, values)


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
