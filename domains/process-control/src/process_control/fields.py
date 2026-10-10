from dataclasses import dataclass
from enum import StrEnum


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
