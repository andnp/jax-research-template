from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.demand_consumption import DemandConsumptionParams, compute_consumption
from process_control.transport import Transport


@dataclass(frozen=True)
class ContactBasinParams:
    total_volume: float
    n_segments: int
    tau: float
    bulk_decay_rate: float = 0.1


@jax_dataclass
class ContactBasinState:
    segments: jax.Array

    @staticmethod
    def create(n_segments: int) -> "ContactBasinState":
        return ContactBasinState(segments=jnp.zeros((n_segments, 5)))


def reset(params: ContactBasinParams, rng_key: jax.Array) -> ContactBasinState:
    return ContactBasinState.create(params.n_segments)


def step(
    state: ContactBasinState,
    inputs: Transport,
    params: ContactBasinParams,
    dt: jax.Array,
    rng_key: jax.Array,
) -> tuple[ContactBasinState, jax.Array]:
    segments = state.segments
    chlorine = segments[:, 0]
    demand_col = segments[:, 1]
    ammonia_col = segments[:, 2]
    turbidity_col = segments[:, 3]
    organics_col = segments[:, 4]

    consumption_params = DemandConsumptionParams(tau=params.tau, bulk_decay_rate=params.bulk_decay_rate)
    new_chlorine, new_demand, demand_consumed = compute_consumption(chlorine, demand_col, consumption_params, dt)

    # Only the ammonia-derived share of the demand pool oxidises ammonia; the
    # rest of demand_consumed is organic/turbidity demand and must not be
    # charged against ammonia. Breakpoint stoichiometry is 7.6:1 Cl2:NH3-N by
    # weight in clean water (practical plants target 8-10:1 for side
    # reactions); see White's Handbook of Chlorination & Alternative
    # Disinfectants, ch. 4, and EPA/WHO breakpoint chlorination guidance.
    ammonia_demand = ammonia_col * 7.6
    ammonia_fraction = jnp.where(demand_col > 0.0, ammonia_demand / demand_col, 0.0)
    ammonia_consumed = jnp.minimum(demand_consumed * ammonia_fraction / 7.6, ammonia_col)
    new_ammonia = jnp.maximum(ammonia_col - ammonia_consumed, 0.0)

    segments = jnp.stack([new_chlorine, new_demand, new_ammonia, turbidity_col, organics_col], axis=1)

    element_volume = params.total_volume / params.n_segments
    flow_volume = inputs.hydraulics.flow * dt
    ratio = jnp.clip(flow_volume / element_volume, 0.0, 1.0)

    outlet_residual = segments[-1, 0]

    inlet_values = jnp.array(
        [
            inputs.composition.chlorine_residual,
            inputs.composition.demand,
            inputs.composition.ammonia,
            inputs.composition.turbidity,
            inputs.composition.organics,
        ]
    )

    downstream = ratio * segments[:-1] + (1.0 - ratio) * segments[1:]
    first_segment = ratio * inlet_values + (1.0 - ratio) * segments[0]
    new_segments = jnp.concatenate([first_segment[None, :], downstream], axis=0)

    return ContactBasinState(segments=new_segments), outlet_residual
