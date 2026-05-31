"""Reduced-order ASM2d: ASM1 extended with PAO phosphorus removal.

Extends the 13-state ASM1 model with 4 additional states for
biological phosphorus removal by phosphate accumulating organisms (PAOs):

  S_PO4 : soluble ortho-phosphate (mg P/L)
  X_PAO : PAO biomass (mg COD/L)
  X_PHA : cell-internal PHA storage (mg COD/L)
  X_PP  : cell-internal poly-phosphate (mg P/L)

The PAO metabolism follows the Mino model:
  - Anaerobic: VFA uptake → PHA storage + poly-P release (P release)
  - Aerobic: PHA consumption → growth + poly-P uptake (P uptake)
  - Anoxic: PAOs can also denitrify using stored PHA

The 13 ASM1 states are computed identically to the ASM1 module, with
additional P-coupling terms for growth and decay.

State vector (17 components, indices 0–16):
  0  S_I     inert soluble (mg COD/L)
  1  S_S     readily biodegradable substrate (mg COD/L)
  2  X_I     inert particulate (mg COD/L)
  3  X_S     slowly biodegradable substrate (mg COD/L)
  4  X_BH    heterotrophic biomass (mg COD/L)
  5  X_BA    autotrophic biomass (mg COD/L)
  6  X_P     particulate products from decay (mg COD/L)
  7  S_O     dissolved oxygen (mg O₂/L)
  8  S_NO    nitrate + nitrite nitrogen (mg N/L)
  9  S_NH    ammonium + ammonia nitrogen (mg N/L)
  10 S_ND    soluble biodegradable organic nitrogen (mg N/L)
  11 X_ND    particulate biodegradable organic nitrogen (mg N/L)
  12 S_ALK   alkalinity (mol HCO₃⁻/m³)
  13 S_PO4   soluble ortho-phosphate (mg P/L)
  14 X_PAO   PAO biomass (mg COD/L)
  15 X_PHA   cell-internal PHA (mg COD/L)
  16 X_PP    cell-internal poly-phosphate (mg P/L)

Reference:
  Henze et al. (1999). Activated Sludge Models ASM1, ASM2, ASM2d and ASM3.
  IWA Scientific and Technical Report No. 9.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.units.asm1 import ASM1Params

N_COMPONENTS_ASM2D = 17


@dataclass(frozen=True)
class ASM2dParams:
    """Parameters for the PAO extension on top of ASM1."""

    # Base ASM1 parameters (for the 13 original states)
    asm1: ASM1Params = ASM1Params(volume=1333.0)

    # PAO kinetics
    q_pha: float = 3.0  # rate constant for PHA storage (1/d)
    q_pp: float = 1.5  # rate constant for poly-P storage (1/d)
    mu_pao: float = 1.0  # max PAO growth rate (1/d)
    b_pao: float = 0.2  # PAO decay rate (1/d)
    b_pha: float = 0.2  # PHA lysis rate (1/d)
    b_pp: float = 0.2  # poly-P lysis rate (1/d)

    # PAO half-saturation coefficients
    k_s_pao: float = 20.0  # S_S half-sat for PAO VFA uptake (mg COD/L)
    k_o_pao: float = 0.2  # O₂ half-sat for PAO (mg/L)
    k_no_pao: float = 0.5  # NO₃ half-sat for PAO denitrification (mg N/L)
    k_p: float = 0.01  # PO₄ half-sat for PAO P uptake (mg P/L)
    k_pha: float = 0.01  # PHA/PAO ratio half-sat for growth
    k_pp: float = 0.01  # PP/PAO ratio half-sat for storage
    k_max_pp: float = 0.34  # max PP/PAO ratio (mg P/mg COD)

    # PAO stoichiometry
    y_pao: float = 0.625  # PAO yield (mg COD biomass / mg COD PHA)
    i_p_bm: float = 0.02  # P content of biomass (mg P / mg COD)
    i_p_xi: float = 0.01  # P content of X_I (mg P / mg COD)
    f_pp_per_pha: float = 0.4  # mol P released per mol PHA stored (simplified)

    # PAO anoxic denitrification
    eta_pao_no: float = 0.6  # anoxic reduction factor for PAO

    # Reference temperature
    t_ref: float = 15.0


def _monod(s: jax.Array, k: float):
    return s / (k + s)


def _inhibition(s: jax.Array, k: float):
    return k / (k + s)


def reactions_asm2d(state: jax.Array, params: ASM2dParams):
    """Compute reaction rates for all 17 state variables.

    Returns dC/dt vector of shape (17,) from biological reactions only.
    """
    # Unpack ASM1 states
    s_s = state[1]
    x_s = state[3]
    x_bh = state[4]
    x_ba = state[5]
    s_o = state[7]
    s_no = state[8]
    s_nh = state[9]
    s_nd = state[10]
    x_nd = state[11]
    _s_alk = state[12]  # noqa: F841 — unpacked for structural clarity matching ASM1 patterns

    # Unpack ASM2d states
    s_po4 = state[13]
    x_pao = state[14]
    x_pha = state[15]
    x_pp = state[16]

    p = params.asm1

    # ===================== ASM1 REACTIONS (processes 1-8) =====================
    # Process 1: Aerobic growth of heterotrophs
    rho_1 = p.mu_h * _monod(s_s, p.k_s) * _monod(s_o, p.k_o_h) * x_bh

    # Process 2: Anoxic growth of heterotrophs
    rho_2 = p.mu_h * _monod(s_s, p.k_s) * _inhibition(s_o, p.k_o_h) * _monod(s_no, p.k_no) * p.eta_g * x_bh

    # Process 3: Aerobic growth of autotrophs
    rho_3 = p.mu_a * _monod(s_nh, p.k_nh) * _monod(s_o, p.k_o_a) * x_ba

    # Process 4: Decay of heterotrophs
    rho_4 = p.b_h * x_bh

    # Process 5: Decay of autotrophs
    rho_5 = p.b_a * x_ba

    # Process 6: Ammonification
    rho_6 = p.k_a * s_nd * x_bh

    # Process 7: Hydrolysis of X_S
    rho_7 = p.k_h * (x_s / (x_bh + 1e-10)) / (p.k_x + x_s / (x_bh + 1e-10)) * (_monod(s_o, p.k_o_h) + p.eta_h * _inhibition(s_o, p.k_o_h) * _monod(s_no, p.k_no)) * x_bh

    # Process 8: Hydrolysis of X_ND
    rho_8 = rho_7 * (x_nd / (x_s + 1e-10))

    # ASM1 derivatives for states 0-12
    dc = jnp.zeros(N_COMPONENTS_ASM2D)

    # S_S (index 1)
    dc = dc.at[1].add(-rho_1 / p.y_h - rho_2 / p.y_h + rho_7)

    # X_S (index 3)
    dc = dc.at[3].add((1.0 - p.f_p) * rho_4 + (1.0 - p.f_p) * rho_5 - rho_7)

    # X_BH (index 4)
    dc = dc.at[4].add(rho_1 + rho_2 - rho_4)

    # X_BA (index 5)
    dc = dc.at[5].add(rho_3 - rho_5)

    # X_P (index 6)
    dc = dc.at[6].add(p.f_p * rho_4 + p.f_p * rho_5)

    # S_O (index 7)
    dc = dc.at[7].add(-(1.0 - p.y_h) / p.y_h * rho_1 - (4.57 - p.y_a) / p.y_a * rho_3)

    # S_NO (index 8)
    dc = dc.at[8].add(-(1.0 - p.y_h) / (2.86 * p.y_h) * rho_2 + rho_3 / p.y_a)

    # S_NH (index 9)
    dc = dc.at[9].add(-p.i_xb * rho_1 - p.i_xb * rho_2 - (p.i_xb + 1.0 / p.y_a) * rho_3 + rho_6)

    # S_ND (index 10)
    dc = dc.at[10].add(-rho_6 + rho_8)

    # X_ND (index 11)
    dc = dc.at[11].add((p.i_xb - p.f_p * p.i_xp) * rho_4 + (p.i_xb - p.f_p * p.i_xp) * rho_5 - rho_8)

    # S_ALK (index 12)
    dc = dc.at[12].add(-p.i_xb / 14.0 * rho_1 + ((1.0 - p.y_h) / (14.0 * 2.86 * p.y_h) - p.i_xb / 14.0) * rho_2 - (p.i_xb / 14.0 + 1.0 / (7.0 * p.y_a)) * rho_3 + rho_6 / 14.0)

    # ===================== PAO REACTIONS (processes 9-14) =====================

    # Ratio of PHA to PAO biomass
    f_pha = x_pha / (x_pao + 1e-10)
    # Ratio of PP to PAO biomass
    f_pp = x_pp / (x_pao + 1e-10)

    # Process 9: Anaerobic PHA storage (VFA uptake + P release)
    # PAOs take up S_S anaerobically, store as PHA, release P from poly-P
    rho_9 = params.q_pha * _monod(s_s, params.k_s_pao) * _inhibition(s_o, params.k_o_pao) * _inhibition(s_no, params.k_no_pao) * (f_pp / (params.k_pp + f_pp)) * x_pao

    # Process 10: Aerobic poly-P storage (P uptake)
    rho_10 = params.q_pp * _monod(s_po4, params.k_p) * _monod(s_o, params.k_o_pao) * (f_pha / (params.k_pha + f_pha)) * (params.k_max_pp - f_pp) / (params.k_max_pp + 1e-10) * x_pao

    # Process 11: Aerobic growth of PAO on stored PHA
    rho_11 = params.mu_pao * _monod(s_o, params.k_o_pao) * _monod(s_nh, p.k_nh) * _monod(s_po4, params.k_p) * (f_pha / (params.k_pha + f_pha)) * x_pao

    # Process 12: Anoxic growth of PAO
    rho_12 = (
        params.mu_pao
        * _inhibition(s_o, params.k_o_pao)
        * _monod(s_no, params.k_no_pao)
        * (_monod(s_nh, p.k_nh) * _monod(s_po4, params.k_p) * f_pha / (params.k_pha + f_pha))
        * params.eta_pao_no
        * x_pao
    )

    # Process 13: PAO lysis
    rho_13 = params.b_pao * x_pao

    # Process 14: PHA lysis
    rho_14 = params.b_pha * x_pha

    # Process 15: PP lysis (releases P)
    rho_15 = params.b_pp * x_pp

    # ===================== PAO contributions to ASM1 states =====================

    # S_S consumed by anaerobic PHA storage
    dc = dc.at[1].add(-rho_9)

    # X_S receives PAO decay products
    dc = dc.at[3].add((1.0 - p.f_p) * rho_13 + rho_14)

    # X_P from PAO decay
    dc = dc.at[6].add(p.f_p * rho_13)

    # S_O consumed by aerobic PAO growth and PP storage
    dc = dc.at[7].add(-(1.0 - params.y_pao) / params.y_pao * rho_11)

    # S_NO: anoxic PAO denitrifies
    dc = dc.at[8].add(-(1.0 - params.y_pao) / (2.86 * params.y_pao) * rho_12)

    # S_NH consumed by PAO growth
    dc = dc.at[9].add(-params.i_p_bm * rho_11 - params.i_p_bm * rho_12)

    # ===================== ASM2d-specific states =====================

    # S_PO4 (index 13): released anaerobically, taken up aerobically
    dc = dc.at[13].add(
        rho_9 * params.f_pp_per_pha  # P release from anaerobic PHA storage
        - rho_10  # aerobic poly-P storage (P uptake)
        - params.i_p_bm * rho_11  # P incorporated into PAO biomass
        - params.i_p_bm * rho_12  # P for anoxic growth
        + params.i_p_bm * rho_13  # P from PAO decay
        + rho_15  # P from PP lysis
    )

    # X_PAO (index 14)
    dc = dc.at[14].add(rho_11 + rho_12 - rho_13)

    # X_PHA (index 15): stored anaerobically, consumed aerobically
    dc = dc.at[15].add(
        rho_9  # PHA stored from VFA uptake
        - rho_11 / params.y_pao  # PHA consumed for aerobic growth
        - rho_12 / params.y_pao  # PHA consumed for anoxic growth
        - rho_10 * 0.2  # small PHA cost for PP storage
        - rho_14  # PHA lysis
    )

    # X_PP (index 16): released anaerobically, stored aerobically
    dc = dc.at[16].add(
        -rho_9 * params.f_pp_per_pha  # PP released during anaerobic storage
        + rho_10  # PP stored aerobically
        - rho_15  # PP lysis
    )

    return dc


def make_default_influent_asm2d():
    """Default ASM2d influent composition (BSM1 influent + phosphorus)."""
    return jnp.array(
        [
            30.0,  # S_I
            69.5,  # S_S
            51.2,  # X_I
            202.32,  # X_S
            28.17,  # X_BH
            0.0,  # X_BA
            0.0,  # X_P
            0.0,  # S_O (will be set by aeration)
            0.0,  # S_NO
            31.56,  # S_NH
            6.95,  # S_ND
            10.59,  # X_ND
            7.0,  # S_ALK
            6.0,  # S_PO4 — typical municipal P concentration
            0.0,  # X_PAO — none in influent
            0.0,  # X_PHA
            0.0,  # X_PP
        ]
    )
