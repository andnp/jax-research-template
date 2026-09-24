"""Reduced-order anaerobic digestion model (ADM1-inspired).

Simplified from the full 32-state IWA ADM1 to a ~9-state model
that captures the essential dynamics for control:

1. Hydrolysis: X_sub -> S_VFA + X_inert + S_IN (first-order, COD-conserving)
2. Methanogenesis: S_VFA -> biomass + CH4 + CO2 (Monod, COD- and
   carbon-conserving stoichiometry)
3. Gas transfer: dissolved CH4/CO2 -> gas phase (Henry's law, exact
   exponential relaxation — see `_relax_to_gas_equilibrium`)
4. Biomass decay: X_bio -> X_sub (first-order)
5. Alkalinity: consumed by net VFA (acid) accumulation, replenished by
   NH4+ release (AM2-style reduced charge balance)
6. pH derived from the alkalinity/dissolved-CO2 ratio

References:
  Batstone et al. (2002). Anaerobic Digestion Model No. 1 (ADM1).
  IWA Scientific and Technical Report No. 13.
  doi:10.2166/wst.2002.0292
  Bernard, Hadj-Sadok, Dochain, Genovesi & Steyer (2001). Dynamical model
  development and parameter identification for an anaerobic wastewater
  treatment process (AM2) — alkalinity/inorganic-carbon state for
  reduced-order digester monitoring and control.
  Biotechnol. Bioeng. 75(4):424-438. doi:10.1002/bit.10036
  Henry's-law base values (298.15 K) and van't Hoff coefficients follow
  the ADM1/BSM2 default parameter set, e.g. Rosen & Jeppsson (2006)
  "Aspects on ADM1 Implementation within the BSM2 Framework"; reproduced
  in the PyADM1 reference implementation:
  https://github.com/paus-5/ADM1/blob/master/python_codes/Adm1_ode.py
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass

# Inner integration substeps per external call. The external step is ~1h
# (dt~0.04 d) but the Monod growth term is stiff relative to that at
# realistic mu_max/K_S values (explicit Euler over the full step can
# overshoot S_vfa in a single substep); substepping keeps forward Euler
# stable without switching to an implicit solver.
_N_SUBSTEPS = 10

# ADM1/BSM2 Henry's-law reference temperature and gas constant (bar*L/(mol*K)).
_T_BASE_K = 298.15
_GAS_CONSTANT_BAR = 0.083145

# Alkalinity bookkeeping (AM2-style, mg CaCO3-equivalent/L): 1 eq of acid or
# base neutralises 50 g CaCO3-equivalent. VFA is tracked as acetate-COD
# (64 g COD/eq); inorganic-N is tracked as mg-N (14 g N/eq for NH4+).
_ALK_PER_COD = 50.0 / 64.0
_ALK_PER_N = 50.0 / 14.0


@dataclass(frozen=True)
class ADM1Params:
    volume: float = 3400.0  # digester volume (m³)
    headspace: float = 300.0  # gas headspace volume (m³)
    t_ref: float = 35.0  # reference temperature (°C)
    theta_hyd: float = 1.05  # Arrhenius factor for hydrolysis
    theta_met: float = 1.10  # Arrhenius factor for methanogenesis

    # Hydrolysis (first-order, COD-conserving)
    k_hyd: float = 0.25  # hydrolysis rate (1/d) at t_ref
    f_vfa_hyd: float = 0.70  # fraction of hydrolysed COD -> soluble VFA
    f_xi_hyd: float = 0.30  # remaining fraction -> inert particulate COD
    # (ADM1 routes disintegration/hydrolysis products it does not track as
    # VFA to particulate/soluble inert pools rather than discarding them;
    # f_vfa_hyd + f_xi_hyd = 1.0 keeps hydrolysis COD-conserving)
    f_in_hyd: float = 0.05  # inorganic-N release, mg-N per mg-COD hydrolysed
    # (independent of the COD split above — a nitrogen yield, not a COD fraction)

    # Methanogenesis (Monod)
    mu_max: float = 0.30  # max specific growth rate (1/d) at t_ref
    k_s_vfa: float = 200.0  # VFA half-saturation (mg-COD/L)
    y_bio: float = 0.05  # biomass yield (g-COD biomass / g-COD VFA consumed)
    # CH4 gets the COD-conserving remainder (1 - y_bio) of VFA-COD consumed;
    # CO2 follows carbon stoichiometry instead (1 mol CO2 per 64 g-COD
    # acetate degraded, from CH3COO- + H2O -> CH4 + HCO3-) since CO2 carries
    # no COD of its own — see `step()`.

    # pH inhibition on methanogenesis (ADM1 default acetoclastic-methanogen limits)
    ph_ll: float = 6.0  # lower pH limit (complete inhibition)
    ph_ul: float = 7.0  # upper pH limit (no inhibition)

    # Gas transfer (Henry's law, kLa for dissolved -> gas)
    kla_ch4: float = 200.0  # gas transfer coefficient CH4 (1/d)
    kla_co2: float = 200.0  # gas transfer coefficient CO2 (1/d)
    h_ch4_298: float = 0.0014  # Henry constant CH4 (mol/L/atm) at 298.15 K
    h_co2_298: float = 0.035  # Henry constant CO2 (mol/L/atm) at 298.15 K
    dh_r_ch4: float = 14240.0  # van't Hoff coefficient (see `_henry_constant`)
    dh_r_co2: float = 19410.0
    # Fixed nominal headspace composition used only for the Henry's-law
    # equilibrium term below, not the reported (mass-balance-derived)
    # gas_ch4_frac. Feeding back the actual, still-transient composition
    # instead makes the two gases' equilibria couple through a state that
    # is itself derived from them each substep, which is unstable; the
    # equilibrium term is a small correction relative to the production
    # flux it is added to, so a fixed typical split (60% CH4 is mid-range
    # for anaerobic digester biogas, e.g. EPA AgSTAR biogas composition
    # guidance: https://www.epa.gov/agstar/biogas-recovery-systems) is an
    # adequate reduced-order treatment.
    p_ch4_typical: float = 0.6

    # Biomass decay
    k_dec: float = 0.02  # decay rate (1/d) at t_ref

    # Alkalinity (state, see `step()`) — initial value only
    alkalinity_init: float = 5000.0  # total alkalinity (mg CaCO3/L)

    # Constraints
    min_vfa: float = 0.0
    min_x_sub: float = 0.0
    min_x_bio: float = 1.0  # minimum viable biomass (mg-COD/L)
    min_alkalinity: float = 0.0


@jax_dataclass
class ADM1State:
    x_substrate: jax.Array  # particulate substrate (mg-COD/L)
    x_biomass: jax.Array  # methanogenic biomass (mg-COD/L)
    x_inert: jax.Array  # inert particulate COD from hydrolysis remainder (mg-COD/L)
    s_vfa: jax.Array  # total VFA (mg-COD/L)
    s_ch4: jax.Array  # dissolved methane (mg-COD/L)
    s_co2: jax.Array  # dissolved CO2 (mg/L as CO2)
    s_in: jax.Array  # inorganic nitrogen (mg-N/L)
    alkalinity: jax.Array  # total alkalinity (mg CaCO3/L equivalent)
    q_gas: jax.Array  # total biogas flow rate (m³/d)
    gas_ch4_frac: jax.Array  # methane fraction in biogas


def _ph_inhibition(ph: jax.Array, ph_ll: float, ph_ul: float):
    """Hill-function pH inhibition factor (0–1)."""
    return jnp.where(
        ph < ph_ul,
        jnp.exp(-3.0 * ((ph - ph_ul) / (ph_ul - ph_ll)) ** 2),
        1.0,
    )


def _estimate_ph(s_co2: jax.Array, alkalinity: jax.Array):
    """Rough pH estimate from CO2 and alkalinity.

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


def _henry_constant(k_h_298: float, dh_over_r: float, t_celsius: jax.Array):
    """Van't Hoff-corrected Henry's-law constant (mol/(L*atm)) at `t_celsius`.

    ADM1/BSM2 report KH at a 298.15 K (25°C) reference, not the digester's
    operating temperature; using the reference value unmodified at 35°C
    overstates solubility (both gases are less soluble as T rises).
    """
    t_kelvin = t_celsius + 273.15
    return k_h_298 * jnp.exp(
        (-dh_over_r / (_GAS_CONSTANT_BAR * 100.0)) * (1.0 / _T_BASE_K - 1.0 / t_kelvin)
    )


def _relax_to_gas_equilibrium(
    s: jax.Array,
    production_rate: jax.Array,
    kla: float | jax.Array,
    s_eq: jax.Array,
    dt: jax.Array,
):
    """Exact solution of ds/dt = production_rate - kla*(s - s_eq) over `dt`.

    Unconditionally stable for any kla*dt — the explicit-Euler form this
    replaces (`kla * s * dt`) requires kla*dt << 1 to avoid diverging
    (kla=200 1/d and dt~0.04 d gives kla*dt~8), and it ignores Henry's law
    equilibrium entirely, so it can also both overshoot and pull the gas
    volume above what mass balance allows. The mass transferred to gas is
    read off this closed-form balance, not summed separately, so it can
    never disagree with the change in dissolved concentration.
    """
    kla_safe = jnp.maximum(kla, 1e-8)
    drive = s - s_eq - production_rate / kla_safe
    s_new = s_eq + production_rate / kla_safe + drive * jnp.exp(-kla_safe * dt)
    transferred_mass = production_rate * dt - (s_new - s)
    return s_new, transferred_mass


def reset(
    feed_cod: float,
    params: ADM1Params,
    rng_key: jax.Array,
):
    """Initialize digester at approximate steady state."""
    return ADM1State(
        x_substrate=jnp.array(feed_cod * 0.3),  # some undigested substrate
        x_biomass=jnp.array(1500.0),  # healthy biomass
        x_inert=jnp.array(0.0),
        s_vfa=jnp.array(200.0),  # moderate VFA
        s_ch4=jnp.array(50.0),  # near saturation
        s_co2=jnp.array(500.0),  # typical
        s_in=jnp.array(800.0),  # ammonia from protein
        alkalinity=jnp.array(params.alkalinity_init),
        q_gas=jnp.array(1000.0),  # typical biogas
        gas_ch4_frac=jnp.array(0.65),  # typical CH4 fraction
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
    sub_dt = dt / _N_SUBSTEPS

    # Temperature-corrected rates (constant across substeps).
    k_hyd = _temp_correction(params.k_hyd, params.theta_hyd, temperature, params.t_ref)
    mu_max = _temp_correction(params.mu_max, params.theta_met, temperature, params.t_ref)
    k_dec = _temp_correction(params.k_dec, params.theta_hyd, temperature, params.t_ref)
    kh_ch4 = _henry_constant(params.h_ch4_298, params.dh_r_ch4, temperature)
    kh_co2 = _henry_constant(params.h_co2_298, params.dh_r_co2, temperature)
    dilution = q_feed / params.volume

    s = state
    ch4_frac = state.gas_ch4_frac
    ch4_transferred_total = jnp.zeros_like(dt)
    co2_transferred_total = jnp.zeros_like(dt)
    ph = _estimate_ph(state.s_co2, state.alkalinity)

    for _ in range(_N_SUBSTEPS):
        ph = _estimate_ph(s.s_co2, s.alkalinity)
        i_ph = _ph_inhibition(ph, params.ph_ll, params.ph_ul)

        # 1. Hydrolysis: X_sub -> S_VFA + X_inert + S_IN
        r_hyd = k_hyd * s.x_substrate
        d_x_sub_hyd = -r_hyd
        d_s_vfa_hyd = r_hyd * params.f_vfa_hyd
        d_x_inert_hyd = r_hyd * params.f_xi_hyd
        d_s_in_hyd = r_hyd * params.f_in_hyd

        # 2. Methanogenesis: S_VFA -> biomass + CH4 + CO2
        mu = mu_max * s.s_vfa / (params.k_s_vfa + s.s_vfa) * i_ph
        r_growth = mu * s.x_biomass
        r_vfa_consumed = r_growth / params.y_bio
        d_s_vfa_met = -r_vfa_consumed
        d_x_bio_growth = r_growth
        # COD balance: y_bio of consumed COD becomes biomass, the
        # COD-conserving remainder becomes CH4 (CO2 carries no COD).
        d_s_ch4_met = r_vfa_consumed * (1.0 - params.y_bio)
        # Carbon stoichiometry (independent of the COD split above):
        # CH3COO- + H2O -> CH4 + HCO3-, i.e. 1 mol CO2 per 64 g-COD acetate.
        d_s_co2_met = r_vfa_consumed * (44.0 / 64.0)

        # 3. Biomass decay: X_bio -> X_sub
        r_dec = k_dec * s.x_biomass
        d_x_bio_dec = -r_dec
        d_x_sub_dec = r_dec

        # 4. Dilution from feed
        d_x_sub_dil = dilution * (feed_cod - s.x_substrate)
        d_x_inert_dil = -dilution * s.x_inert
        d_s_vfa_dil = -dilution * s.s_vfa
        d_s_ch4_dil = -dilution * s.s_ch4
        d_s_co2_dil = -dilution * s.s_co2
        d_s_in_dil = -dilution * s.s_in
        d_x_bio_dil = -dilution * s.x_biomass
        d_alk_dil = -dilution * s.alkalinity

        # 5. Alkalinity (AM2-style): net VFA (acid) accumulation consumes
        # buffering capacity; NH4+ release from ammonification replenishes it.
        d_alk_reaction = -(d_s_vfa_hyd + d_s_vfa_met) * _ALK_PER_COD + d_s_in_hyd * _ALK_PER_N

        new_x_sub = jnp.maximum(
            params.min_x_sub, s.x_substrate + (d_x_sub_hyd + d_x_sub_dec + d_x_sub_dil) * sub_dt
        )
        new_x_inert = jnp.maximum(0.0, s.x_inert + (d_x_inert_hyd + d_x_inert_dil) * sub_dt)
        new_x_bio = jnp.maximum(
            params.min_x_bio, s.x_biomass + (d_x_bio_growth + d_x_bio_dec + d_x_bio_dil) * sub_dt
        )
        new_s_vfa = jnp.maximum(
            params.min_vfa, s.s_vfa + (d_s_vfa_hyd + d_s_vfa_met + d_s_vfa_dil) * sub_dt
        )
        new_s_in = jnp.maximum(0.0, s.s_in + (d_s_in_hyd + d_s_in_dil) * sub_dt)
        new_alk = jnp.maximum(
            params.min_alkalinity, s.alkalinity + (d_alk_reaction + d_alk_dil) * sub_dt
        )

        # 6. Gas transfer: dissolved -> gas phase, at Henry's-law equilibrium.
        # p_ch4/p_co2 use the fixed nominal split (see ADM1Params.p_ch4_typical),
        # not the still-settling `ch4_frac` below — see that field's docstring.
        s_ch4_eq = kh_ch4 * params.p_ch4_typical * 64000.0  # mol/L -> mg-COD/L (64 g-COD/mol CH4)
        s_co2_eq = kh_co2 * (1.0 - params.p_ch4_typical) * 44000.0  # mol/L -> mg/L (44 g/mol CO2)

        new_s_ch4, transferred_ch4 = _relax_to_gas_equilibrium(
            s.s_ch4, d_s_ch4_met + d_s_ch4_dil, params.kla_ch4, s_ch4_eq, sub_dt
        )
        new_s_co2, transferred_co2 = _relax_to_gas_equilibrium(
            s.s_co2, d_s_co2_met + d_s_co2_dil, params.kla_co2, s_co2_eq, sub_dt
        )
        new_s_ch4 = jnp.maximum(0.0, new_s_ch4)
        new_s_co2 = jnp.maximum(0.0, new_s_co2)
        transferred_ch4 = jnp.maximum(0.0, transferred_ch4)
        transferred_co2 = jnp.maximum(0.0, transferred_co2)

        ch4_frac = jnp.where(
            (transferred_ch4 + transferred_co2) > 0.0,
            transferred_ch4 / (transferred_ch4 + transferred_co2 + 1e-10),
            ch4_frac,
        )
        ch4_transferred_total = ch4_transferred_total + transferred_ch4
        co2_transferred_total = co2_transferred_total + transferred_co2

        s = ADM1State(
            x_substrate=new_x_sub,
            x_biomass=new_x_bio,
            x_inert=new_x_inert,
            s_vfa=new_s_vfa,
            s_ch4=new_s_ch4,
            s_co2=new_s_co2,
            s_in=new_s_in,
            alkalinity=new_alk,
            q_gas=s.q_gas,
            gas_ch4_frac=ch4_frac,
        )

    # Biogas production (volume conversion at STP: mg-COD or mg -> mol -> L -> m³)
    ch4_vol = ch4_transferred_total * params.volume / 1e3 / 64.0 * 22.4
    co2_vol = co2_transferred_total * params.volume / 1e3 / 44.0 * 22.4
    q_biogas = (ch4_vol + co2_vol) / (dt + 1e-10)
    ch4_frac_out = jnp.where(q_biogas > 0.0, ch4_vol / (ch4_vol + co2_vol + 1e-10), ch4_frac)

    new_state = ADM1State(
        x_substrate=s.x_substrate,
        x_biomass=s.x_biomass,
        x_inert=s.x_inert,
        s_vfa=s.s_vfa,
        s_ch4=s.s_ch4,
        s_co2=s.s_co2,
        s_in=s.s_in,
        alkalinity=s.alkalinity,
        q_gas=q_biogas,
        gas_ch4_frac=ch4_frac_out,
    )

    return new_state, q_biogas, ch4_frac_out, ph
