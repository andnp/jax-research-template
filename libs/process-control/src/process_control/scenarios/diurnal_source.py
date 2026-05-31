from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.chemistry.demand_model import compute_demand
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.transport import BulkProperties, Composition, Hydraulics, Transport


@dataclass(frozen=True)
class DiurnalSourceParams:
    mean_flow: float
    diurnal_amplitude: float
    min_flow: float
    max_flow: float
    demand_offset: float
    flow_demand_coefficient: float
    demand_noise_std: float
    drift_scale: float
    steps_per_day: int
    base_ammonia: float = 0.3
    base_organics: float = 2.0
    base_turbidity: float = 5.0


@jax_dataclass
class DiurnalSourceState:
    flow_drift: jax.Array
    demand_drift: jax.Array

    @staticmethod
    def create() -> "DiurnalSourceState":
        return DiurnalSourceState(
            flow_drift=jnp.array(0.0),
            demand_drift=jnp.array(0.0),
        )


def reset(rng_key: jax.Array) -> DiurnalSourceState:
    return DiurnalSourceState.create()


def step(
    state: DiurnalSourceState,
    step_count: jax.Array,
    params: DiurnalSourceParams,
    rng_key: jax.Array,
) -> tuple[DiurnalSourceState, Transport, jax.Array, jax.Array]:
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Flow channel: double-harmonic diurnal waveform (sin + cos)
    t = (step_count % params.steps_per_day) / params.steps_per_day
    diurnal_signal = jnp.sin(2.0 * jnp.pi * t) + jnp.cos(4.0 * jnp.pi * t)

    flow_channel = ChannelParams(
        mean=params.mean_flow, amplitude=params.diurnal_amplitude / 2.0,
        min_value=params.min_flow, max_value=params.max_flow,
        drift_scale=params.drift_scale,
        drift_clip=params.diurnal_amplitude / 2.0,
    )
    new_flow_drift, flow = channel_step(
        state.flow_drift, diurnal_signal, flow_channel, k1,
    )

    # Demand composition (drift-only, no sinusoidal pattern)
    demand_drift_delta = jax.random.normal(k2) * params.drift_scale * 0.1
    new_demand_drift = jnp.clip(state.demand_drift + demand_drift_delta, -0.25, 0.25)

    demand_noise = jax.random.normal(k3) * params.demand_noise_std

    ammonia = params.base_ammonia + new_demand_drift * 0.5
    organics = params.base_organics + params.flow_demand_coefficient * flow
    turbidity = params.base_turbidity + new_demand_drift * 2.0 + demand_noise
    demand = compute_demand(ammonia, organics, turbidity) + params.demand_offset

    new_state = DiurnalSourceState(flow_drift=new_flow_drift, demand_drift=new_demand_drift)
    transport = Transport(
        hydraulics=Hydraulics(flow=flow),
        composition=Composition(
            chlorine_residual=jnp.array(0.0),
            demand=demand,
            ammonia=ammonia,
            turbidity=turbidity,
            organics=organics,
        ),
        bulk_properties=BulkProperties(temperature=jnp.array(20.0)),
    )
    return new_state, transport, flow, demand
