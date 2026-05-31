"""Reduced-order anaerobic digestion model (ADM1-inspired).

Simplified from the full 32-state IWA ADM1 to a ~8-state model
that captures the essential dynamics for control:

1. Hydrolysis: X_sub → S_VFA + S_IN (first-order)
2. Methanogenesis: S_VFA → biomass + CH₄ + CO₂ (Monod)
3. Gas transfer: dissolved CH₄/CO₂ → gas phase (Henry's law)
4. Biomass decay: X_bio → X_sub (first-order)
5. pH derived from alkalinity balance

References:
  Batstone et al. (2002). IWA ADM1: Anaerobic Digestion Model No. 1.
  Simplified following Jeong et al. (2005) reduced-order approach.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class ADM1Params:
    volume: float = 3400.0  # digester volume (m³)
    headspace: float = 300.0  # gas headspace volume (m³)
    t_ref: float = 35.0  # reference temperature (°C)
    theta_hyd: float = 1.05  # Arrhenius factor for hydrolysis
    theta_met: float = 1.10  # Arrhenius factor for methanogenesis

    # Hydrolysis (first-order)
    k_hyd: float = 0.25  # hydrolysis rate (1/d) at t_ref
    f_vfa_hyd: float = 0.70  # fraction of hydrolysed COD → VFA
    f_in_hyd: float = 0.05  # fraction → inorganic nitrogen (per g COD)

    # Methanogenesis (Monod)
    mu_max: float = 0.30  # max specific growth rate (1/d) at t_ref
    k_s_vfa: float = 200.0  # VFA half-saturation (mg-COD/L)
    y_bio: float = 0.05  # biomass yield (g-COD biomass / g-COD VFA)
    f_ch4: float = 0.65  # fraction of VFA-COD converted to CH₄
    f_co2: float = 0.35  # fraction → CO₂

    # pH inhibition on methanogenesis
    ph_ll: float = 6.0  # lower pH limit (complete inhibition)
    ph_ul: float = 7.5  # upper pH limit (no inhibition)

    # Gas transfer (Henry's law, KLa for dissolved → gas)
    kla_ch4: float = 200.0  # gas transfer coefficient CH₄ (1/d)
    kla_co2: float = 200.0  # gas transfer coefficient CO₂ (1/d)
    h_ch4: float = 0.0014  # Henry constant CH₄ (mol/L/atm at 35°C)
    h_co2: float = 0.035  # Henry constant CO₂ (mol/L/atm at 35°C)

    # Biomass decay
    k_dec: float = 0.02  # decay rate (1/d) at t_ref

    # pH / alkalinity
    alkalinity: float = 5000.0  # total alkalinity (mg CaCO₃/L) — roughly constant

    # Constraints
    min_vfa: float = 0.0
    min_x_sub: float = 0.0
    min_x_bio: float = 1.0  # minimum viable biomass (mg-COD/L)


@jax_dataclass
class ADM1State:
    x_substrate: jax.Array  # particulate substrate (mg-COD/L)
    x_biomass: jax.Array  # methanogenic biomass (mg-COD/L)
    s_vfa: jax.Array  # total VFA (mg-COD/L)
    s_ch4: jax.Array  # dissolved methane (mg-COD/L)
    s_co2: jax.Array  # dissolved CO₂ (mg/L as CO₂)
    s_in: jax.Array  # inorganic nitrogen (mg-N/L)
    q_gas: jax.Array  # total biogas flow rate (m³/d)
    gas_ch4_frac: jax.Array  # methane fraction in biogas


def _ph_inhibition(ph: jax.Array, ph_ll: float, ph_ul: float):
    """Hill-function pH inhibition factor (0–1)."""
    return jnp.where(
        ph < ph_ul,
        jnp.exp(-3.0 * ((ph - ph_ul) / (ph_ul - ph_ll)) ** 2),
        1.0,
    )


def _estimate_ph(s_co2: jax.Array, alkalinity: float):
    """Rough pH estimate from CO₂ and alkalinity.

    Simplified carbonate equilibrium. At typical digester conditions:
    pH ≈ 6.35 + log10(alkalinity_eq / (s_co2 / 44 * 1000))
    where alkalinity_eq is in meq/L.
    """
    alk_meq = alkalinity / 50.0  # mg CaCO₃/L → meq/L
    co2_molar = s_co2 / 44.0  # mg/L → mmol/L
    ratio = alk_meq / (co2_molar + 1e-6)
    return jnp.clip(6.35 + jnp.log10(ratio + 1e-6), 5.0, 9.0)


def _temp_correction(rate: float, theta: float, t: jax.Array, t_ref: float):
    """Arrhenius temperature correction."""
    return rate * theta ** (t - t_ref)


def reset(
    feed_cod: float,
    params: ADM1Params,
    rng_key: jax.Array,
):
    """Initialize digester at approximate steady state."""
    return ADM1State(
        x_substrate=jnp.array(feed_cod * 0.3),  # some undigested substrate
        x_biomass=jnp.array(1500.0),  # healthy biomass
        s_vfa=jnp.array(200.0),  # moderate VFA
        s_ch4=jnp.array(50.0),  # near saturation
        s_co2=jnp.array(500.0),  # typical
        s_in=jnp.array(800.0),  # ammonia from protein
        q_gas=jnp.array(1000.0),  # typical biogas
        gas_ch4_frac=jnp.array(0.65),  # typical CH₄ fraction
    )


def step(
    state: ADM1State,
    feed_cod: jax.Array,
    q_feed: jax.Array,
    temperature: jax.Array,
    params: ADM1Params,
    dt: jax.Array,
):
    """Advance digester by one timestep.

    Args:
        feed_cod: influent COD concentration (mg/L)
        q_feed: feed flow rate (m³/d)
        temperature: digester temperature (°C)
        dt: timestep (d)

    Returns:
        (new_state, q_biogas, ch4_fraction, ph)
    """
    # Temperature-corrected rates
    k_hyd = _temp_correction(params.k_hyd, params.theta_hyd, temperature, params.t_ref)
    mu_max = _temp_correction(params.mu_max, params.theta_met, temperature, params.t_ref)
    k_dec = _temp_correction(params.k_dec, params.theta_hyd, temperature, params.t_ref)

    # pH and inhibition
    ph = _estimate_ph(state.s_co2, params.alkalinity)
    i_ph = _ph_inhibition(ph, params.ph_ll, params.ph_ul)

    # 1. Hydrolysis: X_sub → S_VFA + S_IN
    r_hyd = k_hyd * state.x_substrate
    d_x_sub_hyd = -r_hyd
    d_s_vfa_hyd = r_hyd * params.f_vfa_hyd
    d_s_in_hyd = r_hyd * params.f_in_hyd

    # 2. Methanogenesis: S_VFA → biomass + CH₄ + CO₂
    mu = mu_max * state.s_vfa / (params.k_s_vfa + state.s_vfa) * i_ph
    r_growth = mu * state.x_biomass
    d_s_vfa_met = -r_growth / params.y_bio
    d_x_bio_growth = r_growth
    d_s_ch4_met = -d_s_vfa_met * params.f_ch4
    d_s_co2_met = -d_s_vfa_met * params.f_co2 * (44.0 / 64.0)  # COD → CO₂ mass

    # 3. Biomass decay: X_bio → X_sub
    r_dec = k_dec * state.x_biomass
    d_x_bio_dec = -r_dec
    d_x_sub_dec = r_dec

    # 4. Gas transfer: dissolved → gas phase
    r_ch4_gas = params.kla_ch4 * state.s_ch4 * dt
    r_co2_gas = params.kla_co2 * state.s_co2 * 0.1 * dt  # CO₂ has higher solubility

    # 5. Dilution from feed
    dilution = q_feed / params.volume
    d_x_sub_dil = dilution * (feed_cod - state.x_substrate)
    d_s_vfa_dil = -dilution * state.s_vfa
    d_s_ch4_dil = -dilution * state.s_ch4
    d_s_co2_dil = -dilution * state.s_co2
    d_s_in_dil = -dilution * state.s_in
    d_x_bio_dil = -dilution * state.x_biomass

    # Integrate (forward Euler)
    new_x_sub = jnp.maximum(
        params.min_x_sub,
        state.x_substrate + (d_x_sub_hyd + d_x_sub_dec + d_x_sub_dil) * dt,
    )
    new_x_bio = jnp.maximum(
        params.min_x_bio,
        state.x_biomass + (d_x_bio_growth + d_x_bio_dec + d_x_bio_dil) * dt,
    )
    new_s_vfa = jnp.maximum(
        params.min_vfa,
        state.s_vfa + (d_s_vfa_hyd + d_s_vfa_met + d_s_vfa_dil) * dt,
    )
    new_s_ch4 = jnp.maximum(0.0, state.s_ch4 + (d_s_ch4_met + d_s_ch4_dil) * dt - r_ch4_gas)
    new_s_co2 = jnp.maximum(0.0, state.s_co2 + (d_s_co2_met + d_s_co2_dil) * dt - r_co2_gas)
    new_s_in = jnp.maximum(0.0, state.s_in + (d_s_in_hyd + d_s_in_dil) * dt)

    # Biogas production
    ch4_vol = r_ch4_gas * params.volume / 1e3 / 64.0 * 22.4  # mg-COD → L CH₄ at STP → m³
    co2_vol = r_co2_gas * params.volume / 1e3 / 44.0 * 22.4
    q_biogas = (ch4_vol + co2_vol) / (dt + 1e-10)
    ch4_frac = jnp.where(q_biogas > 0.0, ch4_vol / (ch4_vol + co2_vol + 1e-10), 0.65)

    new_state = ADM1State(
        x_substrate=new_x_sub,
        x_biomass=new_x_bio,
        s_vfa=new_s_vfa,
        s_ch4=new_s_ch4,
        s_co2=new_s_co2,
        s_in=new_s_in,
        q_gas=q_biogas,
        gas_ch4_frac=ch4_frac,
    )

    return new_state, q_biogas, ch4_frac, ph
