from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import jax

from process_control._jax_dataclass import jax_dataclass
from process_control.binding import NamedAccess, Requirement
from process_control.components import ComponentBasis, Quantity

OXYGEN_BASIS = ComponentBasis("g/m3", Quantity.SPECIES, "O2")


@jax_dataclass
class OxygenExchangeParams:
    """Temperature correction of kLa as kLa · θ^(T − T_ref), with θ = 1.024 in BSM2."""

    kla_theta: float = 1.024
    kla_reference_temperature: float = 15.0


@dataclass(frozen=True)
class OxygenExchange:
    """Volumetric oxygen transfer from an infinite gas reservoir into the liquid, as a concentration source.

    The source, in g/m³/h, is kLa · θ^(T − T_ref) · (S_O,sat(T) − S_O) for kLa in h⁻¹ and T in °C.
    """

    saturation: Callable[[jax.Array], jax.Array]

    access: ClassVar[NamedAccess] = NamedAccess(reads=(Requirement("O2", OXYGEN_BASIS),), writes=(Requirement("O2", OXYGEN_BASIS),))

    def source(self, oxygen: jax.Array, kla: jax.Array, temperature: jax.Array, params: OxygenExchangeParams) -> jax.Array:
        corrected = kla * params.kla_theta ** (temperature - params.kla_reference_temperature)
        return corrected * (self.saturation(temperature) - oxygen)
