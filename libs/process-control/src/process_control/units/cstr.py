from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class CSTRParams:
    volume: float  # tank volume (same units as flow * dt)


@jax_dataclass
class CSTRState:
    concentration: jax.Array  # well-mixed concentration of tracked species

    @staticmethod
    def create() -> "CSTRState":
        return CSTRState(concentration=jnp.array(0.0))


def reset(rng_key: jax.Array) -> CSTRState:
    return CSTRState.create()


def step(
    state: CSTRState,
    inlet_flow: jax.Array,
    inlet_concentration: jax.Array,
    addition_rate: jax.Array,
    params: CSTRParams,
    dt: jax.Array,
) -> CSTRState:
    """Advance CSTR by one time step using a constant-volume mass balance.

    Assumes inlet_flow == outlet_flow (incompressible, constant volume).
    addition_rate is an additional source/sink term in concentration * volume / time units
    (e.g., mol/min when volume is in L and time is in minutes).
    """
    tau = params.volume / jnp.maximum(inlet_flow, 1e-6)
    dC = (inlet_concentration - state.concentration) / tau + addition_rate / params.volume
    return CSTRState(concentration=state.concentration + dC * dt)
