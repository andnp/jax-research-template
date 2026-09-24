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

    The balance dC/dt = (Cin - C)/tau + addition_rate/volume is linear in C,
    so it is integrated with its exact solution C_ss + (C - C_ss)*exp(-dt/tau)
    rather than an explicit-Euler step, which is unconditionally stable
    regardless of how large dt is relative to tau (explicit Euler diverges
    once dt exceeds 2*tau).
    """
    safe_flow = jnp.maximum(inlet_flow, 1e-6)
    tau = params.volume / safe_flow
    steady_state = inlet_concentration + addition_rate / safe_flow
    decay = jnp.exp(-dt / tau)
    new_concentration = steady_state + (state.concentration - steady_state) * decay
    return CSTRState(concentration=new_concentration)
