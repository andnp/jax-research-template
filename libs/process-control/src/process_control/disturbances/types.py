import jax

from process_control.chemistry.demand_model import compute_demand
from process_control.transport import Composition, Hydraulics, Transport

DISTURBANCE_NONE = 0
DISTURBANCE_DEMAND_SLUG = 1
DISTURBANCE_RAIN_STORM = 2


def _no_op(transport: Transport) -> Transport:
    return transport


def demand_slug(transport: Transport, magnitude: jax.Array) -> Transport:
    ammonia = transport.composition.ammonia + magnitude * 0.1
    turbidity = transport.composition.turbidity + magnitude * 2.0
    organics = transport.composition.organics + magnitude * 0.5
    new_composition = Composition(
        chlorine_residual=transport.composition.chlorine_residual,
        demand=compute_demand(ammonia, organics, turbidity),
        ammonia=ammonia,
        turbidity=turbidity,
        organics=organics,
    )
    return Transport(
        hydraulics=transport.hydraulics,
        composition=new_composition,
        bulk_properties=transport.bulk_properties,
    )


def rain_storm(transport: Transport, magnitude: jax.Array) -> Transport:
    flow = transport.hydraulics.flow
    added_flow = magnitude * 10.0
    new_flow = flow + added_flow
    safe_new_flow = jax.numpy.where(new_flow == 0.0, 1.0, new_flow)
    dilution_factor = jax.numpy.where(new_flow == 0.0, 1.0, flow / safe_new_flow)
    new_hydraulics = Hydraulics(flow=new_flow)
    ammonia = transport.composition.ammonia * dilution_factor + magnitude * 0.2
    turbidity = transport.composition.turbidity + magnitude * 15.0
    organics = transport.composition.organics * dilution_factor + magnitude * 0.3
    new_composition = Composition(
        chlorine_residual=transport.composition.chlorine_residual * dilution_factor,
        demand=compute_demand(ammonia, organics, turbidity),
        ammonia=ammonia,
        turbidity=turbidity,
        organics=organics,
    )
    return Transport(
        hydraulics=new_hydraulics,
        composition=new_composition,
        bulk_properties=transport.bulk_properties,
    )


def apply_disturbance_type(transport: Transport, type_id: jax.Array, magnitude: jax.Array) -> Transport:
    branches = [
        _no_op,
        lambda t: demand_slug(t, magnitude),
        lambda t: rain_storm(t, magnitude),
    ]
    return jax.lax.switch(type_id, branches, transport)
