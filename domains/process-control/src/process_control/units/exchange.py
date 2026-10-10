from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import jax

from process_control._jax_dataclass import jax_dataclass
from process_control.binding import Binding, NamedAccess, Requirement
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.kinetics import ZERO_CELSIUS

OXYGEN_BASIS = ComponentBasis("g/m3", Quantity.SPECIES, "O2")


class ExchangeModel[P](Protocol):
    """Volumetric exchange with an external reservoir, as concentration sources of the declared writes per hour.

    It reads its declared channels and named input signals; no reservoir outlet is resolved.
    """

    @property
    def access(self) -> NamedAccess: ...

    @property
    def signals(self) -> tuple[str, ...]: ...

    def rates(self, reads: jax.Array, signals: Mapping[str, jax.Array], temperature_k: jax.Array, params: P) -> jax.Array: ...


@jax_dataclass
class OxygenExchangeParams:
    """Temperature correction of kLa as kLa · θ^(T − T_ref), with θ = 1.024 in BSM2 and T_ref in °C."""

    kla_theta: float | jax.Array = 1.024
    kla_reference_temperature: float | jax.Array = 15.0


@dataclass(frozen=True)
class OxygenExchange:
    """Volumetric oxygen transfer from an infinite gas reservoir into the liquid, as a concentration source.

    The source, in g/m³/h, is kLa · θ^(T − T_ref) · (S_O,sat(T) − S_O) for the `kla` signal in h⁻¹ and
    T in °C. `saturation` takes °C.
    """

    saturation: Callable[[jax.Array], jax.Array]

    @property
    def access(self) -> NamedAccess:
        return NamedAccess(reads=(Requirement("O2", OXYGEN_BASIS),), writes=(Requirement("O2", OXYGEN_BASIS),))

    @property
    def signals(self) -> tuple[str, ...]:
        return ("kla",)

    def rates(self, reads: jax.Array, signals: Mapping[str, jax.Array], temperature_k: jax.Array, params: OxygenExchangeParams) -> jax.Array:
        temperature = temperature_k - ZERO_CELSIUS
        corrected = signals["kla"] * params.kla_theta ** (temperature - params.kla_reference_temperature)
        return corrected * (self.saturation(temperature) - reads)


@dataclass(frozen=True)
class ExchangeSource[P]:
    """An exchange bound to a carrier schema, with its own parameters as the only pytree data."""

    model: ExchangeModel[P]
    schema: ComponentSchema
    binding: Binding
    params: P

    @classmethod
    def bind(cls, model: ExchangeModel[P], params: P, schema: ComponentSchema, channel_map: Mapping[str, str] | None = None) -> "ExchangeSource[P]":
        return cls(model, schema, Binding.resolve(model.access, schema, channel_map), params)

    @property
    def signals(self) -> tuple[str, ...]:
        return self.model.signals

    def add_to(self, total: jax.Array, state: jax.Array, signals: Mapping[str, jax.Array], temperature_k: jax.Array) -> jax.Array:
        return self.binding.scatter_add(total, self.model.rates(self.binding.gather(state), signals, temperature_k, self.params))


jax.tree_util.register_dataclass(ExchangeSource, data_fields=["params"], meta_fields=["model", "schema", "binding"])
