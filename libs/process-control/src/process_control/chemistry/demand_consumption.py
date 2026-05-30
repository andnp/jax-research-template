from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DemandConsumptionParams:
    tau: float


def compute_consumption(chlorine: jax.Array, demand: jax.Array, params: DemandConsumptionParams, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
    useful = jnp.minimum(chlorine, demand)
    consumed = useful / params.tau * dt
    new_chlorine = jnp.maximum(chlorine - consumed, 0.0)
    new_demand = jnp.maximum(demand - consumed, 0.0)
    return new_chlorine, new_demand
