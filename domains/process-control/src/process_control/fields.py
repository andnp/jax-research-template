from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp

from process_control.components import ComponentSchema, Quantity

CONTEXT_TEMPERATURE = "temperature_k"


class Phase(StrEnum):
    LIQUID = "liquid"
    GAS = "gas"


class StreamField(StrEnum):
    FLOW = "flow"
    COMPONENTS = "components"
    TEMPERATURE = "temperature_k"
    PRESSURE = "pressure_pa"


@dataclass(frozen=True)
class LocalField:
    """A field of one of a unit's own ports."""

    port: str
    field: StreamField

    def __str__(self) -> str:
        return f"{self.port}.{self.field}"


@dataclass(frozen=True)
class FieldRef:
    """A field of a port of a named flowsheet node."""

    node: str
    port: str
    field: StreamField

    def __str__(self) -> str:
        return f"{self.node}.{self.port}.{self.field}"


@dataclass(frozen=True)
class PortSpec:
    """Static description of a port: its component schema and phase."""

    schema: ComponentSchema
    phase: Phase

    def __post_init__(self) -> None:
        if self.phase is not Phase.GAS:
            return
        coordinates = zip(self.schema.names, self.schema.bases, strict=True)
        invalid = [name for name, basis in coordinates if basis.quantity is not Quantity.SPECIES or basis.unit != "mol/mol"]
        if invalid:
            raise ValueError(f"gas components must be molecular species in mol/mol, got {invalid}")


@dataclass(frozen=True)
class EvaluationGroup:
    """Output fields solved together, with exactly the inlet fields they read.

    `reads_state` and `input_names` mark dependencies on state and plant inputs, which are available at the
    start of every resolution and never order groups. `diagnostic_names` label the extra arrays a group returns.
    """

    name: str
    outputs: tuple[LocalField, ...]
    reads: tuple[LocalField, ...]
    input_names: tuple[str, ...] = ()
    reads_state: bool = False
    diagnostic_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupResult:
    """Output values in the order of the group's outputs, plus diagnostics in the order of its diagnostic names."""

    values: tuple[jax.Array, ...]
    diagnostics: tuple[jax.Array, ...] = ()


jax.tree_util.register_dataclass(GroupResult, data_fields=["values", "diagnostics"], meta_fields=[])


@dataclass(frozen=True)
class PrescribedContext:
    """Isothermal, constant-pressure outlets: temperature (K) read from an input signal and a fixed absolute pressure (Pa).

    The context is a boundary condition, not a solved energy or pressure balance, so it reads no inlet field.
    """

    pressure_pa: float

    def group(self, outlets: Sequence[str]) -> EvaluationGroup:
        context = (StreamField.TEMPERATURE, StreamField.PRESSURE)
        return EvaluationGroup(
            "context",
            outputs=tuple(LocalField(port, field) for port in outlets for field in context),
            reads=(),
            input_names=(CONTEXT_TEMPERATURE,),
        )

    def evaluate(self, outlets: Sequence[str], inputs: Mapping[str, jax.Array]) -> GroupResult:
        return GroupResult((inputs[CONTEXT_TEMPERATURE], jnp.asarray(self.pressure_pa)) * len(outlets))


def validate_groups(
    groups: Sequence[EvaluationGroup],
    inlets: Mapping[str, PortSpec],
    outlets: Mapping[str, PortSpec] | None = None,
) -> None:
    """Check a unit's group declarations are consistent; with `outlets`, also that every outlet field is owned."""
    _require_unique("group names", (group.name for group in groups))
    _require_unique("diagnostic names", (name for group in groups for name in group.diagnostic_names))
    _require_unique("outputs across groups", (output for group in groups for output in group.outputs))

    for group in groups:
        if not group.outputs:
            raise ValueError(f"group {group.name!r} owns no outputs")
        _require_unique(f"input names of group {group.name!r}", group.input_names)
        unknown = sorted({read.port for read in group.reads if read.port not in inlets})
        if unknown:
            raise ValueError(f"group {group.name!r} reads undeclared inlet ports {unknown}")

    if outlets is None:
        return

    owned = {output for group in groups for output in group.outputs}
    unknown_outputs = sorted(str(output) for output in owned if output.port not in outlets)
    if unknown_outputs:
        raise ValueError(f"groups own fields of undeclared outlet ports {unknown_outputs}")

    expected = {LocalField(port, field) for port in outlets for field in StreamField}
    missing = sorted(str(output) for output in expected - owned)
    if missing:
        raise ValueError(f"outlet fields owned by no group: {missing}")


def _require_unique(label: str, items: Iterable[object]) -> None:
    duplicates = sorted(str(item) for item, count in Counter(items).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} must be unique, duplicated: {duplicates}")
