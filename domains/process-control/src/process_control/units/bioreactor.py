from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.components import ComponentSchema
from process_control.kinetics import KineticsModel
from process_control.streams import Stream
from process_control.units.asm1 import bsm1_oxygen_saturation
from process_control.units.contract import Outlet


@dataclass(frozen=True)
class BioreactorParams[P]:
    """Reactor volume (m³), kinetics parameters, and aeration temperature settings.

    kLa is corrected as kLa · θ^(T − T_ref), with θ = 1.024 in BSM2. `temperature` (°C) applies only
    when the reactor reads no temperature input.
    """

    volume: float
    kinetics: P
    kla_theta: float = 1.024
    kla_reference_temperature: float = 15.0
    temperature: float = 15.0


@dataclass(frozen=True)
class AeratedBioreactor[P]:
    """Constant-volume CSTR with pluggable kinetics, aerated through the kinetics' oxygen component.

    The outlet passes the inlet flow through and carries the reactor state as its composition. Inputs
    are `kla` (h⁻¹) and, when `temperature_input` is set, `temperature` (°C).
    """

    kinetics: KineticsModel[P]
    temperature_input: bool = True
    oxygen_saturation: Callable[[jax.Array], jax.Array] = bsm1_oxygen_saturation

    @property
    def inlet_ports(self) -> Mapping[str, ComponentSchema]:
        return {"in": self.kinetics.schema}

    @property
    def outlet_ports(self) -> Mapping[str, Outlet]:
        return {"out": Outlet(self.kinetics.schema, flow_from=frozenset({"in"}))}

    @property
    def input_ports(self) -> tuple[str, ...]:
        return ("kla", "temperature") if self.temperature_input else ("kla",)

    def derivatives(self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: BioreactorParams[P]) -> jax.Array:
        feed = inlets["in"]
        temperature = inputs["temperature"] if self.temperature_input else jnp.asarray(params.temperature)
        oxygen = self.kinetics.schema.index(self.kinetics.oxygen)

        kla = inputs["kla"] * params.kla_theta ** (temperature - params.kla_reference_temperature)
        transfer = kla * (self.oxygen_saturation(temperature) - state[oxygen])
        reactions = self.kinetics.rates(state, self.kinetics.at_temperature(params.kinetics, temperature))
        return feed.flow / params.volume * (feed.components - state) + reactions.at[oxygen].add(transfer)

    def outlet_flows(
        self, state: jax.Array, inlet_flows: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array], params: BioreactorParams[P]
    ) -> Mapping[str, jax.Array]:
        return {"out": inlet_flows["in"]}

    def outlet_compositions(
        self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: BioreactorParams[P]
    ) -> Mapping[str, jax.Array]:
        return {"out": state}

    def project(self, state: jax.Array, params: BioreactorParams[P]) -> jax.Array:
        return jnp.maximum(state, 0.0)
