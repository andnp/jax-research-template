from typing import Protocol

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@jax_dataclass
class Hydraulics:
    flow: jax.Array

    @staticmethod
    def create(flow: float) -> "Hydraulics":
        return Hydraulics(flow=jnp.array(flow))


@jax_dataclass
class Composition:
    chlorine_residual: jax.Array
    demand: jax.Array
    ammonia: jax.Array
    turbidity: jax.Array
    organics: jax.Array

    @staticmethod
    def create(
        chlorine_residual: float,
        demand: float,
        ammonia: float = 0.0,
        turbidity: float = 0.0,
        organics: float = 0.0,
    ) -> "Composition":
        return Composition(
            chlorine_residual=jnp.array(chlorine_residual),
            demand=jnp.array(demand),
            ammonia=jnp.array(ammonia),
            turbidity=jnp.array(turbidity),
            organics=jnp.array(organics),
        )


@jax_dataclass
class BulkProperties:
    temperature: jax.Array

    @staticmethod
    def create(temperature: float) -> "BulkProperties":
        return BulkProperties(temperature=jnp.array(temperature))


@jax_dataclass
class Transport:
    hydraulics: Hydraulics
    composition: Composition
    bulk_properties: BulkProperties

    @staticmethod
    def create(
        flow: float,
        chlorine_residual: float,
        demand: float,
        temperature: float = 20.0,
        ammonia: float = 0.0,
        turbidity: float = 0.0,
        organics: float = 0.0,
    ) -> "Transport":
        return Transport(
            hydraulics=Hydraulics.create(flow),
            composition=Composition.create(chlorine_residual, demand, ammonia, turbidity, organics),
            bulk_properties=BulkProperties.create(temperature),
        )


class SupportsFlow(Protocol):
    @property
    def flow(self) -> jax.Array: ...


class SupportsResidual(Protocol):
    @property
    def chlorine_residual(self) -> jax.Array: ...
