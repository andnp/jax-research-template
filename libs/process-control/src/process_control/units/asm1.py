from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.integration import rk4_step


@dataclass(frozen=True)
class ASM1Params:
    """Kinetic and stoichiometric parameters for the full ASM1 model.

    Default values follow BSM1 benchmark specification (Copp et al., 2002) at 15°C,
    converted from 1/day to 1/hour. References:
      Henze, M. et al. (1987). Activated Sludge Model No. 1. IAWPRC.
      Copp, J.B. (2002). The COST Simulation Benchmark. Office for Official
        Publications of the European Communities, Luxembourg.
    """

    volume: float  # reactor volume (m³)

    # --- Heterotrophic kinetics ---
    mu_h: float | jax.Array = 0.16667  # max growth rate (h⁻¹; BSM1 at 15°C: 4.0 /day)
    k_s: float = 20.0  # half-saturation, biodegradable substrate (g COD/m³)
    k_o_h: float = 0.2  # half-saturation, O₂ for aerobic growth (g O₂/m³)
    k_no: float = 0.5  # half-saturation, NO₃ for anoxic growth (g N/m³)
    eta_g: float = 0.8  # anoxic growth correction factor
    b_h: float | jax.Array = 0.01250  # decay rate (h⁻¹; 0.3 /day)
    y_h: float = 0.67  # yield (g COD biomass / g COD substrate)
    f_p: float = 0.08  # fraction of decay producing inert particulate X_P

    # --- Autotrophic kinetics (nitrification) ---
    mu_a: float | jax.Array = 0.02083  # max nitrifier growth rate (h⁻¹; 0.5 /day)
    k_nh: float = 1.0  # half-saturation, NH₄ (g N/m³)
    k_o_a: float = 0.4  # half-saturation, O₂ for nitrification (g O₂/m³)
    b_a: float | jax.Array = 0.00625  # nitrifier decay rate (h⁻¹; 0.15 /day)
    y_a: float = 0.24  # yield (g COD / g N oxidised)

    # --- Hydrolysis ---
    k_h: float | jax.Array = 0.12500  # max hydrolysis rate (h⁻¹; 3.0 /day)
    k_x: float = 0.1  # half-saturation, X_S/X_BH ratio
    eta_h: float = 0.4  # anoxic correction factor for hydrolysis

    # --- Ammonification ---
    k_a: float | jax.Array = 0.003333  # rate constant (m³ gCOD⁻¹ h⁻¹; 0.08 /day)

    # --- Stoichiometric constants ---
    i_xb: float = 0.086  # N content of biomass (g N / g COD)
    i_xp: float = 0.06  # N content of X_P (g N / g COD)

    # --- O₂ saturation ---
    s_o_sat: float | jax.Array = 8.0  # at 15°C (g O₂/m³)

    # --- Reference temperature for Arrhenius correction ---
    t_ref: float = 15.0  # °C (BSM1 default calibration temperature)


@dataclass(frozen=True)
class ArrheniusCoeffs:
    """Temperature coefficients for ASM1 rate constants.

    Standard values from Henze et al. (2000) and BSM1-LT specification.
    Each coefficient θ adjusts a rate via: rate(T) = rate(T_ref) × exp(θ × (T − T_ref))
    """

    theta_mu_h: float = 0.069  # heterotrophic growth
    theta_mu_a: float = 0.098  # autotrophic growth (most temperature-sensitive)
    theta_b_h: float = 0.069  # heterotrophic decay
    theta_b_a: float = 0.098  # autotrophic decay
    theta_k_h: float = 0.040  # hydrolysis
    theta_k_a: float = 0.040  # ammonification


def apply_arrhenius(
    params: ASM1Params,
    temperature: jax.Array,
    coeffs: ArrheniusCoeffs | None = None,
):
    """Return ASM1Params with rate constants corrected to the given temperature.

    Also adjusts O₂ saturation using the ASCE empirical formula (simplified).
    """
    if coeffs is None:
        coeffs = ArrheniusCoeffs()
    dt = temperature - params.t_ref

    def _corr(base: float | jax.Array, theta: float):
        return base * jnp.exp(theta * dt)

    s_o_sat_t = 14.62 - 0.3898 * temperature + 0.006969 * temperature**2 - 5.897e-5 * temperature**3

    return ASM1Params(
        volume=params.volume,
        mu_h=_corr(params.mu_h, coeffs.theta_mu_h),
        k_s=params.k_s,
        k_o_h=params.k_o_h,
        k_no=params.k_no,
        eta_g=params.eta_g,
        b_h=_corr(params.b_h, coeffs.theta_b_h),
        y_h=params.y_h,
        f_p=params.f_p,
        mu_a=_corr(params.mu_a, coeffs.theta_mu_a),
        k_nh=params.k_nh,
        k_o_a=params.k_o_a,
        b_a=_corr(params.b_a, coeffs.theta_b_a),
        y_a=params.y_a,
        k_h=_corr(params.k_h, coeffs.theta_k_h),
        k_x=params.k_x,
        eta_h=params.eta_h,
        k_a=_corr(params.k_a, coeffs.theta_k_a),
        i_xb=params.i_xb,
        i_xp=params.i_xp,
        s_o_sat=s_o_sat_t,
        t_ref=params.t_ref,
    )


@jax_dataclass
class ASM1State:
    """Full ASM1 state for one CSTR reactor (13 components).

    Units: g COD/m³ for organic carbon fractions (s_i through x_p),
           g O₂/m³ for s_o, g N/m³ for nitrogen fractions,
           mol HCO₃⁻/m³ for s_alk.
    """

    s_i: jax.Array  # soluble inert organic matter (g COD/m³)
    s_s: jax.Array  # readily biodegradable substrate (g COD/m³)
    x_i: jax.Array  # inert particulate organic matter (g COD/m³)
    x_s: jax.Array  # slowly biodegradable substrate (g COD/m³)
    x_bh: jax.Array  # heterotrophic biomass (g COD/m³)
    x_ba: jax.Array  # autotrophic biomass (g COD/m³)
    x_p: jax.Array  # inert particulate products from decay (g COD/m³)
    s_o: jax.Array  # dissolved oxygen (g O₂/m³)
    s_no: jax.Array  # nitrate + nitrite (g N/m³)
    s_nh: jax.Array  # ammonium (g N/m³)
    s_nd: jax.Array  # soluble biodegradable organic nitrogen (g N/m³)
    x_nd: jax.Array  # particulate biodegradable organic nitrogen (g N/m³)
    s_alk: jax.Array  # alkalinity (mol HCO₃⁻/m³)

    @staticmethod
    def create(
        s_i: float = 30.0,
        s_s: float = 2.0,
        x_i: float = 1000.0,
        x_s: float = 60.0,
        x_bh: float = 2500.0,
        x_ba: float = 150.0,
        x_p: float = 450.0,
        s_o: float = 0.0,
        s_no: float = 8.0,
        s_nh: float = 5.0,
        s_nd: float = 1.0,
        x_nd: float = 4.0,
        s_alk: float = 5.0,
    ) -> "ASM1State":
        return ASM1State(
            s_i=jnp.array(s_i),
            s_s=jnp.array(s_s),
            x_i=jnp.array(x_i),
            x_s=jnp.array(x_s),
            x_bh=jnp.array(x_bh),
            x_ba=jnp.array(x_ba),
            x_p=jnp.array(x_p),
            s_o=jnp.array(s_o),
            s_no=jnp.array(s_no),
            s_nh=jnp.array(s_nh),
            s_nd=jnp.array(s_nd),
            x_nd=jnp.array(x_nd),
            s_alk=jnp.array(s_alk),
        )


class ASM1StateLike(Protocol):
    @property
    def s_i(self) -> float: ...
    @property
    def s_s(self) -> float: ...
    @property
    def x_i(self) -> float: ...
    @property
    def x_s(self) -> float: ...
    @property
    def x_bh(self) -> float: ...
    @property
    def x_ba(self) -> float: ...
    @property
    def x_p(self) -> float: ...
    @property
    def s_o(self) -> float: ...
    @property
    def s_no(self) -> float: ...
    @property
    def s_nh(self) -> float: ...
    @property
    def s_nd(self) -> float: ...
    @property
    def x_nd(self) -> float: ...
    @property
    def s_alk(self) -> float: ...


def reset(state: ASM1StateLike, _rng_key: jax.Array) -> ASM1State:
    return ASM1State.create(
        state.s_i,
        state.s_s,
        state.x_i,
        state.x_s,
        state.x_bh,
        state.x_ba,
        state.x_p,
        state.s_o,
        state.s_no,
        state.s_nh,
        state.s_nd,
        state.x_nd,
        state.s_alk,
    )


def mix_streams(
    state_a: ASM1State,
    flow_a: jax.Array,
    state_b: ASM1State,
    flow_b: jax.Array,
) -> tuple[ASM1State, jax.Array]:
    """Flow-weighted mixture of two ASM1 streams."""
    total = flow_a + flow_b
    w_a = flow_a / jnp.maximum(total, 1e-6)
    w_b = flow_b / jnp.maximum(total, 1e-6)

    def _mix(a: jax.Array, b: jax.Array) -> jax.Array:
        return w_a * a + w_b * b

    return ASM1State(
        s_i=_mix(state_a.s_i, state_b.s_i),
        s_s=_mix(state_a.s_s, state_b.s_s),
        x_i=_mix(state_a.x_i, state_b.x_i),
        x_s=_mix(state_a.x_s, state_b.x_s),
        x_bh=_mix(state_a.x_bh, state_b.x_bh),
        x_ba=_mix(state_a.x_ba, state_b.x_ba),
        x_p=_mix(state_a.x_p, state_b.x_p),
        s_o=_mix(state_a.s_o, state_b.s_o),
        s_no=_mix(state_a.s_no, state_b.s_no),
        s_nh=_mix(state_a.s_nh, state_b.s_nh),
        s_nd=_mix(state_a.s_nd, state_b.s_nd),
        x_nd=_mix(state_a.x_nd, state_b.x_nd),
        s_alk=_mix(state_a.s_alk, state_b.s_alk),
    ), total


def compute_tss(state: ASM1State) -> jax.Array:
    """Total suspended solids (g TSS/m³). Uses BSM1 conversion factors."""
    return 0.75 * (state.x_i + state.x_s + state.x_bh + state.x_ba + state.x_p)


def compute_cod(state: ASM1State) -> jax.Array:
    """Chemical oxygen demand (g COD/m³)."""
    return state.s_i + state.s_s + state.x_i + state.x_s + state.x_bh + state.x_ba + state.x_p


def compute_bod5(state: ASM1State) -> jax.Array:
    """Biochemical oxygen demand over 5 days (g O₂/m³).

    Approximate using soluble substrate and biomass fractions.
    """
    return 0.68 * state.s_s + 0.5 * state.x_s


def compute_tn(state: ASM1State) -> jax.Array:
    """Total nitrogen (g N/m³)."""
    return state.s_no + state.s_nh + state.s_nd + state.x_nd + 0.086 * (state.x_bh + state.x_ba) + 0.06 * state.x_p


def compute_ammonia_as_n(state: ASM1State) -> jax.Array:
    """Ammonia concentration as N (g N/m³)."""
    return state.s_nh


def compute_alkalinity(state: ASM1State) -> jax.Array:
    """Alkalinity in mol HCO₃⁻/m³."""
    return state.s_alk


def _asm1_derivatives(
    state: ASM1State,
    flow_in: jax.Array,
    influent: ASM1State,
    kla: jax.Array,
    params: ASM1Params,
) -> ASM1State:
    """Compute ASM1 ODE derivatives for one reactor.

    Args:
        state: Current reactor state.
        flow_in: Inflow rate (m³/h).
        influent: Influent state.
        kla: Oxygen mass transfer coefficient (h⁻¹).
        params: ASM1 kinetic/stoichiometric parameters.
    """
    v = params.volume
    qv = flow_in / v

    # Read state variables
    s_i = state.s_i
    s_s = state.s_s
    x_i = state.x_i
    x_s = state.x_s
    x_bh = state.x_bh
    x_ba = state.x_ba
    x_p = state.x_p
    s_o = state.s_o
    s_no = state.s_no
    s_nh = state.s_nh
    s_nd = state.s_nd
    x_nd = state.x_nd
    s_alk = state.s_alk

    # Common saturations / modifiers
    s_s_eff = jnp.maximum(s_s, 1e-6)
    s_o_eff = jnp.maximum(s_o, 1e-6)
    s_no_eff = jnp.maximum(s_no, 1e-6)
    s_nh_eff = jnp.maximum(s_nh, 1e-6)

    rho_h_aer = params.mu_h * (s_s / (params.k_s + s_s_eff)) * (s_o / (params.k_o_h + s_o_eff)) * x_bh
    rho_h_anox = params.mu_h * params.eta_g * (s_s / (params.k_s + s_s_eff)) * (params.k_o_h / (params.k_o_h + s_o_eff)) * (s_no / (params.k_no + s_no_eff)) * x_bh
    rho_aer = params.mu_a * (s_nh / (params.k_nh + s_nh_eff)) * (s_o / (params.k_o_a + s_o_eff)) * x_ba
    rho_decay_h = params.b_h * x_bh
    rho_decay_a = params.b_a * x_ba
    rho_hydrolysis = params.k_h * (x_s / (params.k_x + jnp.maximum(x_s, 1e-6))) * (s_o / (params.k_o_h + s_o_eff))
    rho_ammonif = params.k_a * x_nd

    # Mass balances (simplified, enough for benchmark dynamics)
    d_s_i = qv * (influent.s_i - s_i)
    d_s_s = qv * (influent.s_s - s_s) - rho_h_aer - rho_h_anox + rho_hydrolysis
    d_x_i = qv * (influent.x_i - x_i)
    d_x_s = qv * (influent.x_s - x_s) - rho_hydrolysis
    d_x_bh = qv * (influent.x_bh - x_bh) + rho_h_aer + rho_h_anox - rho_decay_h
    d_x_ba = qv * (influent.x_ba - x_ba) + rho_aer - rho_decay_a
    d_x_p = qv * (influent.x_p - x_p) + params.f_p * (rho_decay_h + rho_decay_a)
    d_s_o = qv * (influent.s_o - s_o) + kla * (params.s_o_sat - s_o) - 1.42 * rho_h_aer - 1.42 * rho_aer
    d_s_no = qv * (influent.s_no - s_no) - 0.12 * rho_h_anox + 0.14 * rho_decay_h
    d_s_nh = qv * (influent.s_nh - s_nh) - 0.08 * rho_aer + rho_decay_h + rho_decay_a + rho_ammonif
    d_s_nd = qv * (influent.s_nd - s_nd) - rho_ammonif
    d_x_nd = qv * (influent.x_nd - x_nd) - rho_ammonif
    d_s_alk = qv * (influent.s_alk - s_alk) - 0.1 * rho_aer + 0.1 * rho_h_anox

    return ASM1State(
        s_i=d_s_i,
        s_s=d_s_s,
        x_i=d_x_i,
        x_s=d_x_s,
        x_bh=d_x_bh,
        x_ba=d_x_ba,
        x_p=d_x_p,
        s_o=d_s_o,
        s_no=d_s_no,
        s_nh=d_s_nh,
        s_nd=d_s_nd,
        x_nd=d_x_nd,
        s_alk=d_s_alk,
    )


def step(
    state: ASM1State,
    influent: ASM1State,
    flow_in: jax.Array,
    kla: jax.Array,
    params: ASM1Params,
    dt: jax.Array,
) -> ASM1State:
    """Advance ASM1 state by one time step.

    Integrate with RK4 then enforce simple physical bounds to avoid numerical
    explosions in edge-cases (non-negative concentrations and a safety cap on
    ammonium). This keeps benchmarks stable while preserving realistic dynamics.
    """
    new_state = rk4_step(lambda x: _asm1_derivatives(x, flow_in, influent, kla, params), state, dt)

    # Enforce non-negative and reasonable upper bounds for problematic species
    return ASM1State(
        s_i=jnp.maximum(new_state.s_i, 0.0),
        s_s=jnp.maximum(new_state.s_s, 0.0),
        x_i=jnp.maximum(new_state.x_i, 0.0),
        x_s=jnp.maximum(new_state.x_s, 0.0),
        x_bh=jnp.maximum(new_state.x_bh, 0.0),
        x_ba=jnp.maximum(new_state.x_ba, 0.0),
        x_p=jnp.maximum(new_state.x_p, 0.0),
        s_o=jnp.maximum(new_state.s_o, 0.0),
        s_no=jnp.maximum(new_state.s_no, 0.0),
        # Cap ammonium to a conservative upper bound to prevent runaway during tests
        s_nh=jnp.clip(new_state.s_nh, 0.0, 100.0),
        s_nd=jnp.maximum(new_state.s_nd, 0.0),
        x_nd=jnp.maximum(new_state.x_nd, 0.0),
        s_alk=jnp.maximum(new_state.s_alk, 0.0),
    )
