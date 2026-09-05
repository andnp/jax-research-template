from dataclasses import replace

import jax

from process_control._jax_dataclass import jax_dataclass
from process_control.transport import Transport


@jax_dataclass
class MixerState:
    pass


def reset(rng_key: jax.Array) -> MixerState:
    return MixerState()


def step(state: MixerState, incoming_transport: Transport, dose: jax.Array, dt: jax.Array, rng_key: jax.Array) -> tuple[MixerState, Transport]:
    new_composition = replace(
        incoming_transport.composition,
        chlorine_residual=incoming_transport.composition.chlorine_residual + dose,
    )
    new_transport = replace(incoming_transport, composition=new_composition)
    return state, new_transport
