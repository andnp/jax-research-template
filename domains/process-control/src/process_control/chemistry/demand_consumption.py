from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DemandConsumptionParams:
    tau: float
    # Slow first-order bulk decay applied to whatever chlorine remains once
    # the fast demand pool is satisfied. Field-fit bulk decay coefficients
    # cluster around 0.02-0.2 h^-1 in distribution-system studies (Powell et
    # al., "Factors which control bulk chlorine decay rates", Water Research
    # 34(1), 2000, https://doi.org/10.1016/S0043-1354(99)00140-6); reduced-
    # order default, not calibrated to a specific source water.
    bulk_decay_rate: float = 0.1


def compute_consumption(
    chlorine: jax.Array,
    demand: jax.Array,
    params: DemandConsumptionParams,
    dt: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Two-phase chlorine loss: fast demand exertion, then slow bulk decay.

    Operator-split approximation of the parallel first-order decay form
    (Haas & Karra, "Kinetics of wastewater chlorine demand exertion",
    J. WPCF 56(2), 1984): a fast pool (the immediate organic/turbidity/
    ammonia demand) is exhausted at rate 1/tau, then any remaining chlorine
    decays slowly via bulk reaction at `bulk_decay_rate`. Both phases use the
    exact exponential solution of their linear ODE rather than an explicit
    Euler step, so the result is stable for any dt/tau ratio and never goes
    negative. Returns (new_chlorine, new_demand, demand_consumed), where
    demand_consumed is the fast-phase draw only (excludes bulk decay), for
    callers that need to attribute the demand-driven loss to its source.
    """
    decay_factor = jnp.exp(-dt / params.tau)
    demand_limited = chlorine < demand

    # chlorine >= demand: demand pool empties exponentially; chlorine loses exactly that much.
    demand_after_excess = demand * decay_factor
    chlorine_after_excess = chlorine - (demand - demand_after_excess)

    # chlorine < demand: chlorine is the limiting reagent and decays exponentially itself.
    chlorine_after_limited = chlorine * decay_factor
    demand_after_limited = demand - (chlorine - chlorine_after_limited)

    chlorine_after_demand = jnp.maximum(jnp.where(demand_limited, chlorine_after_limited, chlorine_after_excess), 0.0)
    demand_after = jnp.maximum(jnp.where(demand_limited, demand_after_limited, demand_after_excess), 0.0)
    demand_consumed = chlorine - chlorine_after_demand

    new_chlorine = chlorine_after_demand * jnp.exp(-params.bulk_decay_rate * dt)
    return new_chlorine, demand_after, demand_consumed
