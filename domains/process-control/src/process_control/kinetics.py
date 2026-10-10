from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import jax

from process_control.binding import Binding, NamedAccess
from process_control.components import ComponentSchema

ZERO_CELSIUS = 273.15  # K


class KineticsModel[P](Protocol):
    """Reaction source rates over the module's locally declared channels."""

    @property
    def access(self) -> NamedAccess: ...

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: P) -> jax.Array:
        """Rates of `access.writes` per hour in their declared bases, from `access.reads` at a temperature in K.

        `params` hold reference values; any temperature correction happens here, once per call.
        """
        ...


@dataclass(frozen=True)
class KineticSource[P]:
    """Kinetics bound to a carrier schema, with its own parameters as the only pytree data."""

    model: KineticsModel[P]
    schema: ComponentSchema
    binding: Binding
    params: P

    signals: tuple[str, ...] = ()

    @classmethod
    def bind(cls, model: KineticsModel[P], params: P, schema: ComponentSchema, channel_map: Mapping[str, str] | None = None) -> "KineticSource[P]":
        return cls(model, schema, Binding.resolve(model.access, schema, channel_map), params)

    def add_to(self, total: jax.Array, state: jax.Array, signals: Mapping[str, jax.Array], temperature_k: jax.Array) -> jax.Array:
        return self.binding.scatter_add(total, self.model.rates(self.binding.gather(state), temperature_k, self.params))


jax.tree_util.register_dataclass(KineticSource, data_fields=["params"], meta_fields=["model", "schema", "binding", "signals"])
