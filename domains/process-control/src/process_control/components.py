import math
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp


class Quantity(StrEnum):
    SPECIES = "species"
    ELEMENT = "element"
    COD = "cod"
    EQUIVALENT = "equivalent"
    PROXY = "proxy"


@dataclass(frozen=True)
class ComponentBasis:
    """What a component channel's value measures: its unit, kind of quantity and reference species, element or convention."""

    unit: str
    quantity: Quantity
    reference: str

    def __post_init__(self) -> None:
        if not self.unit or not self.reference:
            raise ValueError(f"basis unit and reference must be nonempty, got {self!r}")


@dataclass(frozen=True)
class ComponentSchema:
    """Static description of a component vector: names, particulate flags and COD-to-TSS factors."""

    names: tuple[str, ...]
    particulate: tuple[bool, ...]
    tss_factor: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"component names must be unique: {self.names}")
        if not len(self.names) == len(self.particulate) == len(self.tss_factor):
            raise ValueError("names, particulate and tss_factor must have equal lengths")
        for name, particulate, factor in zip(self.names, self.particulate, self.tss_factor, strict=True):
            if not math.isfinite(factor) or factor < 0:
                raise ValueError(f"TSS factor of {name!r} must be finite and non-negative, got {factor}")
            if factor and not particulate:
                raise ValueError(f"soluble component {name!r} cannot contribute to TSS")

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def particulate_mask(self) -> jax.Array:
        return jnp.array(self.particulate)

    @property
    def soluble_mask(self) -> jax.Array:
        return ~self.particulate_mask

    def tss(self, components: jax.Array) -> jax.Array:
        return components @ jnp.array(self.tss_factor)
