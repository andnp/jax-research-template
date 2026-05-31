"""Membrane filtration unit model with fouling dynamics.

Models a low-pressure membrane (MF/UF) with:
- Reversible fouling (removed by backwash)
- Irreversible fouling (slow accumulation, requires CIP)
- Transmembrane pressure (TMP) that rises with fouling
- Air scour reduces fouling rate
- Backwash restores reversible permeability

The resistance-in-series model:
    TMP = J * mu * (R_m + R_rev + R_irr)
where J = flux (m/s), mu = viscosity, R = resistance.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class MembraneParams:
    area: float = 100.0  # membrane area (m²)
    r_membrane: float = 1e11  # clean membrane resistance (1/m)
    mu: float = 1.002e-3  # water viscosity at 20°C (Pa·s)
    max_tmp: float = 2.0e5  # max TMP before damage (Pa) — ~2 bar

    # Fouling rates
    k_rev_fouling: float = 5e8  # reversible fouling rate (1/m per g/m³ per h)
    k_irr_fouling: float = 1e6  # irreversible fouling rate (1/m per g/m³ per h)
    air_scour_factor: float = 0.6  # fouling rate reduction at max air scour

    # Backwash
    bw_recovery: float = 0.95  # fraction of reversible fouling removed per backwash
    bw_duration: float = 0.005  # backwash duration (h) — ~18s
    bw_flux_ratio: float = 2.0  # backwash flux as multiple of operating flux

    # Feed quality → permeate quality
    rejection: float = 0.999  # TSS rejection (log-4 removal)

    # CIP (chemical in place cleaning)
    cip_recovery: float = 0.80  # fraction of irreversible fouling removed per CIP
    cip_interval: float = 720.0  # hours between CIPs (~30 days)


@jax_dataclass
class MembraneState:
    r_reversible: jax.Array  # reversible fouling resistance (1/m)
    r_irreversible: jax.Array  # irreversible fouling resistance (1/m)
    hours_since_bw: jax.Array  # hours since last backwash
    hours_since_cip: jax.Array  # hours since last CIP
    permeate_volume: jax.Array  # cumulative permeate (m³)


def reset(params: MembraneParams, rng_key: jax.Array):
    return MembraneState(
        r_reversible=jnp.array(0.0),
        r_irreversible=jnp.array(0.0),
        hours_since_bw=jnp.array(0.0),
        hours_since_cip=jnp.array(0.0),
        permeate_volume=jnp.array(0.0),
    )


def compute_tmp(
    flux: jax.Array,
    state: MembraneState,
    params: MembraneParams,
):
    """Compute transmembrane pressure (Pa) for given flux."""
    r_total = params.r_membrane + state.r_reversible + state.r_irreversible
    return flux * params.mu * r_total


def step(
    state: MembraneState,
    feed_tss: jax.Array,
    flux: jax.Array,
    air_scour: jax.Array,
    do_backwash: jax.Array,
    params: MembraneParams,
    dt: jax.Array,
):
    """Advance membrane by one timestep.

    Args:
        feed_tss: feed TSS (g/m³)
        flux: permeate flux setpoint (m/h) — converted internally to m/s
        air_scour: normalised air scour intensity (0-1)
        do_backwash: boolean trigger (1.0 = backwash this step)
        dt: timestep (h)

    Returns:
        (new_state, tmp, permeate_tss, q_permeate)
    """
    # Air scour reduces fouling rate
    scour_effect = 1.0 - params.air_scour_factor * jnp.clip(air_scour, 0.0, 1.0)

    # Fouling accumulation
    d_r_rev = params.k_rev_fouling * feed_tss * scour_effect * dt
    d_r_irr = params.k_irr_fouling * feed_tss * scour_effect * dt

    new_r_rev = state.r_reversible + d_r_rev
    new_r_irr = state.r_irreversible + d_r_irr

    # Backwash: remove reversible fouling
    new_r_rev = jnp.where(
        do_backwash > 0.5,
        new_r_rev * (1.0 - params.bw_recovery),
        new_r_rev,
    )

    # Automatic CIP at interval
    do_cip = (state.hours_since_cip + dt) >= params.cip_interval
    new_r_irr = jnp.where(
        do_cip,
        new_r_irr * (1.0 - params.cip_recovery),
        new_r_irr,
    )

    # Permeate production (during backwash, flux is reversed → no permeate)
    effective_flux = jnp.where(do_backwash > 0.5, 0.0, flux)
    q_permeate = effective_flux * params.area  # m³/h

    # TMP at current conditions
    flux_ms = effective_flux / 3600.0  # m/h → m/s
    r_total = params.r_membrane + new_r_rev + new_r_irr
    tmp = flux_ms * params.mu * r_total

    # Permeate quality
    permeate_tss = feed_tss * (1.0 - params.rejection)

    new_state = MembraneState(
        r_reversible=new_r_rev,
        r_irreversible=new_r_irr,
        hours_since_bw=jnp.where(do_backwash > 0.5, jnp.array(0.0), state.hours_since_bw + dt),
        hours_since_cip=jnp.where(do_cip, jnp.array(0.0), state.hours_since_cip + dt),
        permeate_volume=state.permeate_volume + q_permeate * dt,
    )

    return new_state, tmp, permeate_tss, q_permeate
