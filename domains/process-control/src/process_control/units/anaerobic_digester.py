"""Reduced-order anaerobic digestion model (ADM1-inspired).

Simplified from the full 32-state IWA ADM1 to a ~10-state model
that captures the essential dynamics for control:

1. Hydrolysis: X_sub -> S_VFA + X_inert + S_IN (first-order, COD-conserving)
2. Methanogenesis: S_VFA -> biomass + CH4 + CO2 (Monod, COD-conserving;
   CH4:CO2 split follows Buswell substrate stoichiometry, not pure acetate)
3. Carbonate speciation: dissolved inorganic carbon (S_IC) splits into
   CO2(aq) and HCO3- via the pKa1/alkalinity equilibrium; only CO2(aq) is
   subject to gas transfer (`_speciate_inorganic_carbon`). The whole S_IC
   pool (both species) dilutes with the effluent — alkalinity (net
   charge) and S_IC (carbon mass) are separately conserved quantities,
   each with its own dilution term, not one double-counted loss.
4. Gas transfer: dissolved CH4 / CO2(aq) -> gas phase (Henry's law, exact
   exponential relaxation — see `_relax_to_gas_equilibrium`)
5. Biomass decay: X_bio -> X_sub (first-order)
6. Alkalinity: consumed by net VFA (acid) accumulation, replenished by
   NH4+ release (AM2-style reduced charge balance)
7. pH derived from the HCO3-/CO2(aq) ratio (Henderson-Hasselbalch)

References:
  Batstone et al. (2002). Anaerobic Digestion Model No. 1 (ADM1).
  IWA Scientific and Technical Report No. 13.
  doi:10.2166/wst.2002.0292
  Bernard, Hadj-Sadok, Dochain, Genovesi & Steyer (2001). Dynamical model
  development and parameter identification for an anaerobic wastewater
  treatment process (AM2) — alkalinity/inorganic-carbon state for
  reduced-order digester monitoring and control.
  Biotechnol. Bioeng. 75(4):424-438. doi:10.1002/bit.10036
  Henry's-law base values (298.15 K), van't Hoff coefficients and the
  pKa1(CO2) temperature coefficient follow the ADM1/BSM2 default parameter
  set — Rosen & Jeppsson (2006) "Aspects on ADM1 Implementation within
  the BSM2 Framework"; verified against the Java port of that Matlab
  reference implementation (DAEModel.java, `kLco2`, `K_H_ch4`, `K_H_co2`,
  `K_a_co2` temperature terms): https://github.com/liampetti/jADM1
  Buswell & Mueller (1952). Mechanism of methane fermentation.
  Ind. Eng. Chem. 44(3):550-552. doi:10.1021/ie50507a033 — CH4:CO2 ratio
  from substrate elemental (C,H,O,N) composition; reports 60-65% CH4
  as typical for sludge-digestion systems of this type.
  Metcalf & Eddy / Tchobanoglous, Burton & Stensel (2003), Wastewater
  Engineering, 4th ed., Ch.4 — primary sludge approximated as C10H19O3N,
  the default composite formula used below (`_SUBSTRATE_C/H/O/N`), which
  also fixes the substrate's ammonification (N-release) yield: both are
  derived from the same Buswell equation and elemental formula so they
  cannot drift apart independently (see `_SUBSTRATE_CO2_MOL` etc. below).
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

# Apparent first dissociation "constant" of the CO2(aq)/HCO3- carbonate
# equilibrium (lumping true H2CO3 in with CO2(aq), the standard convention
# in wastewater/ADM1 practice), pKa1=6.35 at 25 C, with a van't Hoff
# temperature correction using effective enthalpy 7646 J/mol — both values
# verified against the ADM1/BSM2 reference parameter set (Rosen & Jeppsson
# 2006), specifically `pK_a_co2_base=6.35` and the `7646.0` coefficient in
# `K_a_co2 = 10^-pK_a_co2_base * exp(7646.0*factor)` in the Java port of
# the BSM2 Matlab code: DAEModel.java lines 152/150,
# https://github.com/liampetti/jADM1 (this shift is <0.05 pH units over
# 25-35 C either way).
_PKA1_CO2_298 = 6.35
_DH_PKA1_CO2 = 7646.0  # J/mol, verified — see note above

# Composite substrate elemental formula (C_a H_b O_c N_d), representative of
# municipal primary/WAS sludge: C10H19O3N (Tchobanoglous, Burton & Stensel,
# "Wastewater Engineering", 4th ed., Metcalf & Eddy, 2003, Ch.4). Both the
# CH4:CO2 production ratio (`ADM1Params.co2_ch4_molar_ratio`) and the
# ammonification N-release yield (`ADM1Params.f_in_hyd`) are derived from
# these SAME four numbers via the Buswell equation, so they cannot drift
# out of stoichiometric consistency with each other:
#   CnHaObNc -> CH4*(n/2+a/8-b/4-3c/8) + CO2*(n/2-a/8+b/4+3c/8) + c*NH3
#   (Buswell & Mueller 1952, Ind. Eng. Chem. 44(3):550, doi:10.1021/ie50507a033)
_SUBSTRATE_C = 10.0
_SUBSTRATE_H = 19.0
_SUBSTRATE_O = 3.0
_SUBSTRATE_N = 1.0
_SUBSTRATE_CH4_MOL = (
    _SUBSTRATE_C / 2.0 + _SUBSTRATE_H / 8.0 - _SUBSTRATE_O / 4.0 - 3.0 * _SUBSTRATE_N / 8.0
)  # 6.25 mol CH4 per mol substrate
_SUBSTRATE_CO2_MOL = (
    _SUBSTRATE_C / 2.0 - _SUBSTRATE_H / 8.0 + _SUBSTRATE_O / 4.0 + 3.0 * _SUBSTRATE_N / 8.0
)  # 3.75 mol CO2 per mol substrate
# Theoretical oxygen demand (g O2/mol substrate) — same electron-balance
# identity used to derive Henry-consistent COD, so that dividing the fixed
# N release (c mol N -> 14*c g N) by it gives an N-release-per-COD yield
# consistent with the same elemental formula, not an independently chosen
# number: ThOD = 8*(4a+b-2c-3d) for CaHbOcNd (verified against acetate,
# C2H4O2: ThOD=64 g O2/mol, matching its known 1.067 g O2/g COD factor).
_SUBSTRATE_THOD = 8.0 * (
    4.0 * _SUBSTRATE_C + _SUBSTRATE_H - 2.0 * _SUBSTRATE_O - 3.0 * _SUBSTRATE_N
)  # 400 g O2/mol substrate


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
    f_in_hyd: float = _SUBSTRATE_N * 14.0 / _SUBSTRATE_THOD  # ~0.035 mg-N/mg-COD hydrolysed
    # (independent of the COD split above — a nitrogen yield, not a COD
    # fraction. Ammonification is the digester's main alkalinity SOURCE
    # (`_ALK_PER_N` below). Derived from the SAME C10H19O3N formula as
    # `co2_ch4_molar_ratio` (1 mol NH3, 14 g N, per mol substrate; ThOD
    # 400 g O2/mol substrate — see `_SUBSTRATE_THOD` — gives 14/400=0.035
    # mg N per mg COD), rather than picked independently, so the two
    # cannot drift out of stoichiometric consistency with each other.

    # Methanogenesis (Monod)
    mu_max: float = 0.30  # max specific growth rate (1/d) at t_ref
    k_s_vfa: float = 200.0  # VFA half-saturation (mg-COD/L)
    y_bio: float = 0.05  # biomass yield (g-COD biomass / g-COD VFA consumed)
    # CH4 gets the COD-conserving remainder (1 - y_bio) of VFA-COD consumed
    # (COD is defined as O2-equivalent oxidising capacity, and CH4 is the
    # only anaerobic end product that carries any, so all degraded COD not
    # routed to biomass ends up as CH4-COD regardless of substrate — see
    # `step()`). CO2 carries no COD; its production instead follows the
    # substrate's carbon/oxidation-state stoichiometry, below.

    # CO2:CH4 molar production ratio from the Buswell equation applied to
    # the same C10H19O3N composite substrate formula as `f_in_hyd` above
    # (`_SUBSTRATE_CO2_MOL`/`_SUBSTRATE_CH4_MOL` = 3.75/6.25 = 0.6), giving
    # a produced-gas CH4 fraction of 1/(1+0.6) = 62.5%, within the 60-65%
    # Buswell & Mueller (1952, Ind. Eng. Chem. 44(3):550,
    # doi:10.1021/ie50507a033) report as typical for sludge-digestion
    # systems (vs. 50:50 for a pure-carbohydrate/acetate substrate, i.e.
    # ratio 1.0, used previously). This is a fixed composition, not a
    # state — real sludge composition varies, but the model does not track
    # carbohydrate/protein/lipid fractions separately.
    co2_ch4_molar_ratio: float = _SUBSTRATE_CO2_MOL / _SUBSTRATE_CH4_MOL

    # pH inhibition on methanogenesis (ADM1 default acetoclastic-methanogen limits)
    ph_ll: float = 6.0  # lower pH limit (complete inhibition)
    ph_ul: float = 7.0  # upper pH limit (no inhibition)

    # Gas transfer (Henry's law, kLa for dissolved -> gas). kla_ch4/kla_co2
    # and the van't Hoff coefficients below are verified, not tuned: they
    # match the ADM1/BSM2 reference parameter set exactly (`kLco2=200.0`,
    # `-14240.0*factor` for K_H_ch4, `-19410.0*factor` for K_H_co2, in
    # DAEModel.java/DigesterParameters.java, https://github.com/liampetti/jADM1,
    # a Java port of Rosen & Jeppsson's BSM2 Matlab implementation).
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
    # flux it is added to, so a fixed typical split (62.5% CH4 matches the
    # `co2_ch4_molar_ratio` default above, and is mid-range for anaerobic
    # digester biogas, e.g. EPA AgSTAR biogas composition guidance:
    # https://www.epa.gov/agstar/biogas-recovery-systems) is an adequate
    # reduced-order treatment.
    p_ch4_typical: float = 0.625

    # Biomass decay
    k_dec: float = 0.02  # decay rate (1/d) at t_ref

    # Alkalinity (state, see `step()`) — initial value only
    alkalinity_init: float = 5000.0  # total alkalinity (mg CaCO3/L)
    ic_init: float = 5000.0  # total dissolved inorganic carbon (mg/L as CO2)

    # Constraints
    min_vfa: float = 0.0
    min_x_sub: float = 0.0
    min_x_bio: float = 1.0  # minimum viable biomass (mg-COD/L)
    min_alkalinity: float = 0.0
    min_ic: float = 0.0


@jax_dataclass
class ADM1State:
    x_substrate: jax.Array  # particulate substrate (mg-COD/L)
    x_biomass: jax.Array  # methanogenic biomass (mg-COD/L)
    x_inert: jax.Array  # inert particulate COD from hydrolysis remainder (mg-COD/L)
    s_vfa: jax.Array  # total VFA (mg-COD/L)
    s_ch4: jax.Array  # dissolved methane (mg-COD/L)
    s_ic: jax.Array  # total dissolved inorganic carbon, CO2(aq)+HCO3- (mg/L as CO2)
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


def _pka1_co2(t_celsius: jax.Array):
    """Van't Hoff-corrected apparent pKa1 of the CO2(aq)/HCO3- equilibrium (see `_PKA1_CO2_298`)."""
    t_kelvin = t_celsius + 273.15
    return _PKA1_CO2_298 + (_DH_PKA1_CO2 / (_GAS_CONSTANT_BAR * 100.0 * jnp.log(10.0))) * (
        1.0 / t_kelvin - 1.0 / _T_BASE_K
    )


def _speciate_inorganic_carbon(s_ic: jax.Array, alkalinity: jax.Array, t_celsius: jax.Array):
    """Split total dissolved inorganic carbon into CO2(aq) and HCO3-, and derive pH.

    Alkalinity is taken as the HCO3- pool (mg/L as CO2-equivalent, i.e. same
    carbon-mass basis as `s_ic`) — the AM2-style simplification already used
    for `alkalinity` itself, since VFA- and NH4+ dominate the rest of the
    charge balance and are tracked separately. CO2(aq) is the remainder of
    `s_ic`; pH follows from Henderson-Hasselbalch on their ratio. Only
    CO2(aq) is subject to gas transfer in `step()`; `s_ic` as a whole still
    dilutes with the effluent there (`alkalinity`'s own, separate dilution
    term tracks net charge, not this carbon mass a second time).
    """
    hco3_as_co2 = (alkalinity / 50.0) * 44.0  # mg CaCO3/L -> meq/L -> mg/L as CO2
    co2aq = jnp.maximum(s_ic - hco3_as_co2, 1e-3)
    ratio = hco3_as_co2 / co2aq
    ph = jnp.clip(_pka1_co2(t_celsius) + jnp.log10(ratio + 1e-6), 5.0, 9.0)
    return co2aq, ph


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
        s_ic=jnp.array(params.ic_init),
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
    _, ph = _speciate_inorganic_carbon(state.s_ic, state.alkalinity, temperature)

    for _ in range(_N_SUBSTEPS):
        _, ph = _speciate_inorganic_carbon(s.s_ic, s.alkalinity, temperature)
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
        # Buswell substrate stoichiometry (independent of the COD split
        # above — see ADM1Params.co2_ch4_molar_ratio): CO2 production
        # scales with CH4 production at the substrate's CO2:CH4 molar ratio.
        d_s_ic_met = d_s_ch4_met * params.co2_ch4_molar_ratio * (44.0 / 64.0)

        # 3. Biomass decay: X_bio -> X_sub
        r_dec = k_dec * s.x_biomass
        d_x_bio_dec = -r_dec
        d_x_sub_dec = r_dec

        # 4. Dilution from feed
        d_x_sub_dil = dilution * (feed_cod - s.x_substrate)
        d_x_inert_dil = -dilution * s.x_inert
        d_s_vfa_dil = -dilution * s.s_vfa
        d_s_ch4_dil = -dilution * s.s_ch4
        # The effluent carries ALL dissolved inorganic carbon out (CO2(aq)
        # and HCO3- together — S_IC is one conserved carbon pool), so this
        # term dilutes the whole pool. Alkalinity's own dilution term below
        # (`d_alk_dil`) tracks a DIFFERENT conserved quantity (net charge),
        # not a second copy of this carbon; diluting only the free-CO2(aq)
        # share here (a change briefly tried and reverted) undercounts
        # effluent carbon and overcounts vented CO2, artificially lowering
        # the gas-phase CH4 fraction.
        d_s_ic_dil = -dilution * s.s_ic
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
        # Only CO2(aq) (not HCO3-, which stays in solution as part of
        # alkalinity) is subject to Henry's law, so the S_IC equilibrium
        # target is offset by the bicarbonate pool implied by `new_alk` —
        # this is the mechanism by which CO2 produced by fermentation is
        # retained in solution (as HCO3-, or exported with the effluent)
        # rather than vented, raising the gas-phase CH4 fraction above the
        # raw Buswell production ratio. p_ch4/p_co2 use the fixed nominal
        # split (see ADM1Params.p_ch4_typical), not the still-settling
        # `ch4_frac` below — see that field's docstring.
        hco3_as_co2 = (new_alk / 50.0) * 44.0  # mg CaCO3/L -> meq/L -> mg/L as CO2
        s_ch4_eq = kh_ch4 * params.p_ch4_typical * 64000.0  # mol/L -> mg-COD/L (64 g-COD/mol CH4)
        s_ic_eq = hco3_as_co2 + kh_co2 * (1.0 - params.p_ch4_typical) * 44000.0  # mg/L as CO2

        new_s_ch4, transferred_ch4 = _relax_to_gas_equilibrium(
            s.s_ch4, d_s_ch4_met + d_s_ch4_dil, params.kla_ch4, s_ch4_eq, sub_dt
        )
        new_s_ic, transferred_co2 = _relax_to_gas_equilibrium(
            s.s_ic, d_s_ic_met + d_s_ic_dil, params.kla_co2, s_ic_eq, sub_dt
        )
        new_s_ch4 = jnp.maximum(0.0, new_s_ch4)
        new_s_ic = jnp.maximum(params.min_ic, new_s_ic)
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
            s_ic=new_s_ic,
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
        s_ic=s.s_ic,
        s_in=s.s_in,
        alkalinity=s.alkalinity,
        q_gas=q_biogas,
        gas_ch4_frac=ch4_frac_out,
    )

    return new_state, q_biogas, ch4_frac_out, ph
