"""Primary clarifier unit model.

A simplified gravity settler for primary treatment. Removes TSS based
on surface loading rate (SLR) with a Monod-type saturation:

    eta(SLR) = eta_max * K_SLR / (K_SLR + SLR)

Sludge accumulates at the bottom and must be wasted to prevent
inventory build-up. The underflow TSS concentration depends on
current sludge inventory.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PrimaryClarifierParams:
    area: float = 500.0  # settling area (m²)
    depth: float = 3.0  # settler depth (m)
    eta_max: float = 0.65  # max TSS removal efficiency at low SLR
    k_slr: float = 1.5  # half-saturation SLR (m³/m²/h)
    max_sludge_mass: float = 5e6  # kg — capacity limit (triggers overflow)
    min_underflow_tss: float = 5000.0  # g/m³ minimum underflow concentration


@jax_dataclass
class PrimaryClarifierState:
    sludge_mass: jax.Array  # accumulated sludge (g)


def reset(params: PrimaryClarifierParams, rng_key: jax.Array):
    initial_mass = params.max_sludge_mass * 0.3
    return PrimaryClarifierState(sludge_mass=jnp.array(initial_mass))


def step(
    state: PrimaryClarifierState,
    feed_tss: jax.Array,
    q_feed: jax.Array,
    q_waste: jax.Array,
    params: PrimaryClarifierParams,
    dt: jax.Array,
):
    """Advance primary clarifier by one timestep.

    Args:
        feed_tss: influent TSS (g/m³)
        q_feed: influent flow (m³/h)
        q_waste: primary sludge wastage rate (m³/h)
        dt: timestep (h)

    Returns:
        (new_state, effluent_tss, underflow_tss)
    """
    slr = q_feed / params.area
    eta = params.eta_max * params.k_slr / (params.k_slr + slr)

    tss_removed = eta * feed_tss
    effluent_tss = feed_tss - tss_removed

    # Sludge mass balance
    mass_in = tss_removed * q_feed * dt  # g deposited
    underflow_tss = jnp.maximum(
        state.sludge_mass / (params.area * params.depth) * 2.0,
        params.min_underflow_tss,
    )
    mass_out = underflow_tss * q_waste * dt  # g removed
    new_mass = jnp.clip(state.sludge_mass + mass_in - mass_out, 0.0, params.max_sludge_mass)

    new_state = PrimaryClarifierState(sludge_mass=new_mass)
    return new_state, effluent_tss, underflow_tss
