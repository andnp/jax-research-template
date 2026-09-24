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
    # Measured clean UF/NF membrane resistances run ~1e12-1e14 1/m depending on
    # pore size (tighter membranes resist more): 2.2e13-9.0e13 1/m for 300-8000 Da
    # UF/NF membranes, https://pmc.ncbi.nlm.nih.gov/articles/PMC5618119/. This
    # benchmark models a looser MF/UF membrane, near the low end of that range.
    r_membrane: float = 1e12  # clean membrane resistance (1/m)
    water_temp_c: float = 20.0  # feed/membrane temperature (°C)
    max_tmp: float = 2.0e5  # max TMP before damage (Pa) — ~2 bar

    # Cake fouling: deposited cake mass per unit area grows as J*C*dt (the
    # classical cake-filtration form underlying the Darcy's-law resistance
    # model TMP = J*mu*(R_m + R_cake), e.g. Luongo et al. 2021 eq. 2.1,
    # https://arxiv.org/abs/2104.03996), and each mass increment adds
    # resistance alpha_cake * mass. Reported specific cake resistances for
    # particulate/NOM cakes span ~1e13-1e16 m/kg; alpha_cake below sits near
    # the low (more permeable) end for reduced-order tuning of this benchmark.
    alpha_cake: float = 5e13  # specific cake resistance (m/kg)
    # Fraction of each cake increment that remains after backwash (compaction
    # or pore blocking rather than a loose cake) — reduced-order tuning value,
    # not verified against a specific measured source.
    irr_fraction: float = 0.01
    air_scour_factor: float = 0.6  # fouling rate reduction at max air scour

    # Above the critical flux, fouling accelerates rather than merely scaling
    # with J (Field, Wu, Howell & Gupta 1995, "Critical flux concept for
    # microfiltration fouling", J. Membrane Sci. 100:259-272,
    # https://doi.org/10.1016/0376-7388(94)00265-Z). The gain is a reduced-order
    # tuning value, not itself from that source.
    flux_critical: float = 0.03  # critical flux (m/h)
    flux_critical_gain: float = 2.0  # extra fouling multiplier per m/h above critical

    # Backwash
    bw_recovery: float = 0.95  # fraction of reversible fouling removed per backwash
    bw_duration: float = 0.005  # backwash duration (h) — ~18s
    bw_flux_ratio: float = 2.0  # backwash flux as multiple of operating flux

    # Feed quality → permeate quality
    rejection: float = 0.999  # TSS rejection: -log10(1 - rejection) = 3 (log-3 removal)

    # CIP (chemical in place cleaning)
    cip_recovery: float = 0.80  # fraction of irreversible fouling removed per CIP
    cip_interval: float = 720.0  # hours between CIPs (~30 days)


@jax_dataclass
class MembraneState:
    r_reversible: jax.Array  # reversible fouling resistance (1/m)
    r_irreversible: jax.Array  # irreversible fouling resistance (1/m)
    hours_since_bw: jax.Array  # hours since last backwash
    is_backwashing: jax.Array  # whether a backwash continues into the next step
    backwash_remaining: jax.Array  # remaining backwash time (h)
    backwash_trigger_latched: jax.Array  # prevents a held-high command from retriggering
    hours_since_cip: jax.Array  # hours since last CIP
    permeate_volume: jax.Array  # cumulative permeate (m³)


def reset(params: MembraneParams, rng_key: jax.Array):
    return MembraneState(
        r_reversible=jnp.array(0.0),
        r_irreversible=jnp.array(0.0),
        hours_since_bw=jnp.array(0.0),
        is_backwashing=jnp.array(False),
        backwash_remaining=jnp.array(0.0),
        backwash_trigger_latched=jnp.array(False),
        hours_since_cip=jnp.array(0.0),
        permeate_volume=jnp.array(0.0),
    )


def _water_viscosity(temp_c: jax.Array | float) -> jax.Array:
    """Water dynamic viscosity (Pa·s) from temperature (°C).

    Standard exponential (Vogel-type) fit; matches tabulated values to
    within ~0.5% over 0-100°C (e.g. 1.002 mPa·s at 20°C, 0.890 mPa·s at
    25°C) — https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html
    """
    temp_k = temp_c + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (temp_k - 140.0))


def compute_tmp(
    flux: jax.Array,
    state: MembraneState,
    params: MembraneParams,
):
    """Compute transmembrane pressure (Pa) for given flux."""
    r_total = params.r_membrane + state.r_reversible + state.r_irreversible
    return flux * _water_viscosity(params.water_temp_c) * r_total


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
        do_backwash: rising-edge trigger (1.0 starts one backwash)
        dt: timestep (h)

    Returns:
        (new_state, tmp, permeate_tss, q_permeate)
    """
    requested = do_backwash > 0.5
    rising_edge = requested & ~state.backwash_trigger_latched
    start_backwash = rising_edge & ~state.is_backwashing
    duration = jnp.asarray(params.bw_duration)
    remaining_at_start = jnp.where(start_backwash, duration, state.backwash_remaining)
    backwash_active = state.is_backwashing | start_backwash

    # A backwash may occupy only part of a timestep. Held-high and repeated
    # triggers are ignored until the command first returns low.
    wash_time = jnp.where(backwash_active, jnp.minimum(dt, remaining_at_start), 0.0)
    instant_backwash = start_backwash & (duration <= 0.0)
    operating_time = dt - wash_time
    new_bw_remaining = jnp.maximum(remaining_at_start - wash_time, 0.0)
    new_is_backwashing = backwash_active & (new_bw_remaining > 0.0)

    # Air scour reduces fouling rate while the membrane is producing permeate.
    scour_effect = 1.0 - params.air_scour_factor * jnp.clip(air_scour, 0.0, 1.0)

    # Cake mass deposits as flux * feed concentration * time; resistance
    # grows as alpha_cake * deposited mass, so higher flux fouls faster for
    # the same feed load. Above the critical flux, deposition accelerates
    # rather than merely scaling with flux (see MembraneParams.flux_critical).
    feed_tss_kg_m3 = feed_tss / 1000.0  # g/m³ → kg/m³
    flux_above_critical = jnp.maximum(flux - params.flux_critical, 0.0)
    fouling_flux = flux + params.flux_critical_gain * flux_above_critical
    deposition_rate = params.alpha_cake * fouling_flux * feed_tss_kg_m3 * scour_effect
    d_r_rev = deposition_rate * operating_time
    d_r_irr = params.irr_fraction * d_r_rev

    new_r_rev = state.r_reversible
    new_r_irr = state.r_irreversible + d_r_irr

    # Apply cleaning continuously so total recovery does not depend on dt.
    recovery_fraction = jnp.clip(params.bw_recovery, 0.0, 1.0)
    safe_duration = jnp.maximum(duration, jnp.finfo(wash_time.dtype).tiny)
    wash_fraction = jnp.where(duration > 0.0, wash_time / safe_duration, 0.0)
    recovery_multiplier = (1.0 - recovery_fraction) ** wash_fraction
    recovery_multiplier = jnp.where(instant_backwash, 1.0 - recovery_fraction, recovery_multiplier)
    new_r_rev = new_r_rev * recovery_multiplier + d_r_rev

    # Automatic CIP at interval
    do_cip = (state.hours_since_cip + dt) >= params.cip_interval
    new_r_irr = jnp.where(
        do_cip,
        new_r_irr * (1.0 - params.cip_recovery),
        new_r_irr,
    )

    # Report the timestep-average gross permeate flow. Backwash water use is not
    # subtracted; the membrane simply produces no permeate during wash time.
    safe_dt = jnp.maximum(dt, jnp.finfo(operating_time.dtype).tiny)
    operating_fraction = jnp.where(dt > 0.0, operating_time / safe_dt, 0.0)
    effective_flux = flux * operating_fraction
    q_permeate = effective_flux * params.area  # m³/h

    # TMP at current conditions
    flux_ms = effective_flux / 3600.0  # m/h → m/s
    r_total = params.r_membrane + new_r_rev + new_r_irr
    tmp = flux_ms * _water_viscosity(params.water_temp_c) * r_total

    # Permeate quality
    permeate_tss = feed_tss * (1.0 - params.rejection)

    new_state = MembraneState(
        r_reversible=new_r_rev,
        r_irreversible=new_r_irr,
        hours_since_bw=jnp.where(
            backwash_active | instant_backwash,
            operating_time,
            state.hours_since_bw + dt,
        ),
        is_backwashing=new_is_backwashing,
        backwash_remaining=new_bw_remaining,
        backwash_trigger_latched=requested,
        hours_since_cip=jnp.where(do_cip, jnp.array(0.0), state.hours_since_cip + dt),
        permeate_volume=state.permeate_volume + q_permeate * dt,
    )

    return new_state, tmp, permeate_tss, q_permeate
