from dataclasses import dataclass

import jax
import jax.numpy as jnp


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
    mu_h: float = 0.16667   # max growth rate (h⁻¹; BSM1 at 15°C: 4.0 /day)
    k_s: float = 20.0       # half-saturation, biodegradable substrate (g COD/m³)
    k_o_h: float = 0.2      # half-saturation, O₂ for aerobic growth (g O₂/m³)
    k_no: float = 0.5       # half-saturation, NO₃ for anoxic growth (g N/m³)
    eta_g: float = 0.8      # anoxic growth correction factor
    b_h: float = 0.01250    # decay rate (h⁻¹; 0.3 /day)
    y_h: float = 0.67       # yield (g COD biomass / g COD substrate)
    f_p: float = 0.08       # fraction of decay producing inert particulate X_P

    # --- Autotrophic kinetics (nitrification) ---
    mu_a: float = 0.02083   # max nitrifier growth rate (h⁻¹; 0.5 /day)
    k_nh: float = 1.0       # half-saturation, NH₄ (g N/m³)
    k_o_a: float = 0.4      # half-saturation, O₂ for nitrification (g O₂/m³)
    b_a: float = 0.00625    # nitrifier decay rate (h⁻¹; 0.15 /day)
    y_a: float = 0.24       # yield (g COD / g N oxidised)

    # --- Hydrolysis ---
    k_h: float = 0.12500    # max hydrolysis rate (h⁻¹; 3.0 /day)
    k_x: float = 0.1        # half-saturation, X_S/X_BH ratio
    eta_h: float = 0.4      # anoxic correction factor for hydrolysis

    # --- Ammonification ---
    k_a: float = 0.003333   # rate constant (m³ gCOD⁻¹ h⁻¹; 0.08 /day)

    # --- Stoichiometric constants ---
    i_xb: float = 0.086     # N content of biomass (g N / g COD)
    i_xp: float = 0.06      # N content of X_P (g N / g COD)

    # --- O₂ saturation ---
    s_o_sat: float = 8.0    # at 15°C (g O₂/m³)


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


def reset(
    s_i: float,
    s_s: float,
    x_i: float,
    x_s: float,
    x_bh: float,
    x_ba: float,
    x_p: float,
    s_o: float,
    s_no: float,
    s_nh: float,
    s_nd: float,
    x_nd: float,
    s_alk: float,
    rng_key: jax.Array,
) -> ASM1State:
    return ASM1State.create(s_i, s_s, x_i, x_s, x_bh, x_ba, x_p, s_o, s_no, s_nh, s_nd, x_nd, s_alk)


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
    """Total COD (g COD/m³) — soluble + particulate."""
    return state.s_i + state.s_s + state.x_i + state.x_s + state.x_bh + state.x_ba + state.x_p


def step(
    state: ASM1State,
    inlet: ASM1State,
    inlet_flow: jax.Array,
    kla: jax.Array,
    params: ASM1Params,
    dt: jax.Array,
) -> ASM1State:
    """Advance ASM1 reactor by one time step using forward Euler integration.

    The step implements all 8 ASM1 processes:
      ρ1  aerobic growth of heterotrophs
      ρ2  anoxic growth of heterotrophs (denitrification)
      ρ3  aerobic growth of autotrophs (nitrification)
      ρ4  decay of heterotrophs
      ρ5  decay of autotrophs
      ρ6  ammonification of soluble organic N
      ρ7  hydrolysis of particulate substrate
      ρ8  hydrolysis of particulate organic N

    Non-negativity is enforced after integration (jnp.maximum / jnp.clip).
    kla is the volumetric O₂ transfer rate (h⁻¹); set to 0 for anoxic operation.
    """
    D = inlet_flow / params.volume  # hydraulic dilution rate (h⁻¹)

    # ── Process rates ──────────────────────────────────────────────────────────
    # ρ1: aerobic heterotrophic growth
    r1 = (
        params.mu_h
        * (state.s_s / (params.k_s + state.s_s))
        * (state.s_o / (params.k_o_h + state.s_o))
        * state.x_bh
    )
    # ρ2: anoxic heterotrophic growth (denitrification)
    r2 = (
        params.mu_h
        * params.eta_g
        * (state.s_s / (params.k_s + state.s_s))
        * (params.k_o_h / (params.k_o_h + state.s_o))
        * (state.s_no / (params.k_no + state.s_no))
        * state.x_bh
    )
    # ρ3: autotrophic growth (nitrification)
    r3 = (
        params.mu_a
        * (state.s_nh / (params.k_nh + state.s_nh))
        * (state.s_o / (params.k_o_a + state.s_o))
        * state.x_ba
    )
    # ρ4, ρ5: decay
    r4 = params.b_h * state.x_bh
    r5 = params.b_a * state.x_ba

    # ρ6: ammonification
    r6 = params.k_a * state.s_nd * state.x_bh

    # ρ7: hydrolysis of X_S
    x_s_ratio = (state.x_s / jnp.maximum(state.x_bh, 1e-6))
    hydrolysis_switching = (
        state.s_o / (params.k_o_h + state.s_o)
        + params.eta_h * (params.k_o_h / (params.k_o_h + state.s_o)) * (state.s_no / (params.k_no + state.s_no))
    )
    r7 = params.k_h * (x_s_ratio / (params.k_x + x_s_ratio)) * hydrolysis_switching * state.x_bh

    # ρ8: hydrolysis of X_ND (proportional to X_S hydrolysis)
    r8 = (state.x_nd / jnp.maximum(state.x_s, 1e-6)) * r7

    # ── Mass balances ──────────────────────────────────────────────────────────
    ds_i = D * (inlet.s_i - state.s_i)
    ds_s = D * (inlet.s_s - state.s_s) - (r1 + r2) / params.y_h + r7
    dx_i = D * (inlet.x_i - state.x_i)
    dx_s = D * (inlet.x_s - state.x_s) + (1.0 - params.f_p) * (r4 + r5) - r7
    dx_bh = D * (inlet.x_bh - state.x_bh) + r1 + r2 - r4
    dx_ba = D * (inlet.x_ba - state.x_ba) + r3 - r5
    dx_p = D * (inlet.x_p - state.x_p) + params.f_p * (r4 + r5)
    ds_o = (
        D * (inlet.s_o - state.s_o)
        + kla * (params.s_o_sat - state.s_o)
        - (1.0 - params.y_h) / params.y_h * r1
        - (4.57 - params.y_a) / params.y_a * r3
    )
    ds_no = (
        D * (inlet.s_no - state.s_no)
        - (1.0 - params.y_h) / (2.86 * params.y_h) * r2
        + r3 / params.y_a
    )
    ds_nh = (
        D * (inlet.s_nh - state.s_nh)
        - params.i_xb * (r1 + r2)
        - (params.i_xb + 1.0 / params.y_a) * r3
        + r6
    )
    ds_nd = D * (inlet.s_nd - state.s_nd) - r6 + r8
    dx_nd = (
        D * (inlet.x_nd - state.x_nd)
        + (params.i_xb - params.f_p * params.i_xp) * (r4 + r5)
        - r8
    )
    ds_alk = (
        D * (inlet.s_alk - state.s_alk)
        - params.i_xb / 14.0 * r1
        + ((1.0 - params.y_h) / (2.86 * params.y_h) - params.i_xb) / 14.0 * r2
        - (params.i_xb + 1.0 / (7.0 * params.y_a)) / 14.0 * r3
        + r6 / 14.0
    )

    # ── Euler step with physical bounds ───────────────────────────────────────
    return ASM1State(
        s_i=jnp.maximum(state.s_i + ds_i * dt, 0.0),
        s_s=jnp.maximum(state.s_s + ds_s * dt, 0.0),
        x_i=jnp.maximum(state.x_i + dx_i * dt, 0.0),
        x_s=jnp.maximum(state.x_s + dx_s * dt, 0.0),
        x_bh=jnp.maximum(state.x_bh + dx_bh * dt, 0.0),
        x_ba=jnp.maximum(state.x_ba + dx_ba * dt, 0.0),
        x_p=jnp.maximum(state.x_p + dx_p * dt, 0.0),
        s_o=jnp.clip(state.s_o + ds_o * dt, 0.0, params.s_o_sat),
        s_no=jnp.maximum(state.s_no + ds_no * dt, 0.0),
        s_nh=jnp.maximum(state.s_nh + ds_nh * dt, 0.0),
        s_nd=jnp.maximum(state.s_nd + ds_nd * dt, 0.0),
        x_nd=jnp.maximum(state.x_nd + dx_nd * dt, 0.0),
        s_alk=state.s_alk + ds_alk * dt,  # alkalinity can be negative
    )
