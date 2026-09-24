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
    bulk_decay_rate: float = 0.02


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
    # `chlorine` is total free residual (HOCl + OCl-), not speciated. HOCl is
    # the more effective disinfectant and its share depends on pH (pKa~7.5 at
    # 25C, Morris, J. Phys. Chem. 70(12), 1966); Transport carries no pH
    # signal, so speciation is not modeled here.
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

    # n_segments approximates the T10/T baffling factor from EPA's LT2ESWTR
    # Disinfection Profiling and Benchmarking guidance: more tanks-in-series
    # pushes T10/T closer to 1 (plug flow), fewer segments toward a lower
    # T10/T (more short-circuiting), https://www.epa.gov/system/files/documents/2022-02/disprof_bench_3rules_final_508.pdf
    element_volume = params.total_volume / params.n_segments
    flow_volume = inputs.hydraulics.flow * dt
    # A single tanks-in-series update can only move fluid one segment per
    # call; a ratio > 1 (flow_volume > element_volume) would silently discard
    # mass through the clip below. Substep with lax.fori_loop so every
    # sub-update sees ratio <= 1, however large dt or flow gets.
    n_substeps = jnp.maximum(jnp.ceil(flow_volume / element_volume), 1.0).astype(jnp.int32)
    sub_ratio = jnp.clip(flow_volume / (element_volume * n_substeps.astype(jnp.float32)), 0.0, 1.0)

    inlet_values = jnp.array(
        [
            inputs.composition.chlorine_residual,
            inputs.composition.demand,
            inputs.composition.ammonia,
            inputs.composition.turbidity,
            inputs.composition.organics,
        ]
    )

    def _advect_once(_: jax.Array, segs: jax.Array) -> jax.Array:
        downstream = sub_ratio * segs[:-1] + (1.0 - sub_ratio) * segs[1:]
        first_segment = sub_ratio * inlet_values + (1.0 - sub_ratio) * segs[0]
        return jnp.concatenate([first_segment[None, :], downstream], axis=0)

    new_segments = jax.lax.fori_loop(0, n_substeps, _advect_once, segments)
    # The physically meaningful outlet is what leaves after this step's
    # transport, not the pre-transport last-segment value.
    outlet_residual = new_segments[-1, 0]

    return ContactBasinState(segments=new_segments), outlet_residual
