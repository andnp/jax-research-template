from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BiologicalReactorParams:
    """Kinetic parameters for a simplified ASM1 biological reactor.

    Default values follow BSM1 standard (Henze et al., 2000) converted to
    per-hour units (BSM1 defines rates in per-day).
    """
    volume: float  # m³

    # Heterotrophic kinetics
    mu_h: float = 0.25        # max growth rate (h⁻¹; BSM1: 6.0 /day)
    k_s: float = 20.0         # half-saturation for substrate (g COD/m³)
    k_o_h: float = 0.2        # half-saturation for O₂, aerobic growth (g O₂/m³)
    k_no: float = 0.5         # half-saturation for NO₃, anoxic growth (g N/m³)
    eta_g: float = 0.8        # anoxic growth correction factor
    b_h: float = 0.02583      # decay rate (h⁻¹; BSM1: 0.62 /day)
    y_h: float = 0.67         # heterotrophic yield (g COD/g COD)
    f_p: float = 0.08         # fraction of decay as inert particulate

    # Autotrophic kinetics (nitrification)
    mu_a: float = 0.03333     # max nitrifier growth rate (h⁻¹; BSM1: 0.8 /day)
    k_nh: float = 1.0         # half-saturation for NH₄ (g N/m³)
    k_o_a: float = 0.4        # half-saturation for O₂, nitrification (g O₂/m³)
    b_a: float = 0.00833      # nitrifier decay rate (h⁻¹; BSM1: 0.2 /day)
    y_a: float = 0.24         # autotrophic yield (g COD/g N)

    # Stoichiometric constants
    i_xb: float = 0.086       # N content of biomass (g N/g COD)
    s_o_sat: float = 8.0      # O₂ saturation (g O₂/m³) at 15°C


@dataclass(frozen=True)
class BiologicalReactorState:
    """State of a CSTR biological reactor with simplified ASM1 kinetics.

    All concentrations in g/m³ (equivalent to mg/L).
    """
    s_s: jax.Array    # readily biodegradable substrate (g COD/m³)
    s_o: jax.Array    # dissolved oxygen (g O₂/m³)
    s_no: jax.Array   # nitrate + nitrite (g N/m³)
    s_nh: jax.Array   # ammonium (g N/m³)
    x_bh: jax.Array   # heterotrophic biomass (g COD/m³)
    x_ba: jax.Array   # autotrophic biomass (g COD/m³)

    @staticmethod
    def create(
        s_s: float = 2.0,
        s_o: float = 0.0,
        s_no: float = 5.0,
        s_nh: float = 8.0,
        x_bh: float = 2500.0,
        x_ba: float = 150.0,
    ) -> "BiologicalReactorState":
        return BiologicalReactorState(
            s_s=jnp.array(s_s),
            s_o=jnp.array(s_o),
            s_no=jnp.array(s_no),
            s_nh=jnp.array(s_nh),
            x_bh=jnp.array(x_bh),
            x_ba=jnp.array(x_ba),
        )


jax.tree_util.register_dataclass(
    BiologicalReactorState,
    data_fields=["s_s", "s_o", "s_no", "s_nh", "x_bh", "x_ba"],
    meta_fields=[],
)


def reset(
    s_s: float,
    s_o: float,
    s_no: float,
    s_nh: float,
    x_bh: float,
    x_ba: float,
    rng_key: jax.Array,
) -> BiologicalReactorState:
    return BiologicalReactorState.create(s_s, s_o, s_no, s_nh, x_bh, x_ba)


def mix_streams(
    state_a: BiologicalReactorState,
    flow_a: jax.Array,
    state_b: BiologicalReactorState,
    flow_b: jax.Array,
) -> tuple[BiologicalReactorState, jax.Array]:
    """Flow-weighted mixture of two streams."""
    total_flow = flow_a + flow_b
    w_a = flow_a / jnp.maximum(total_flow, 1e-6)
    w_b = flow_b / jnp.maximum(total_flow, 1e-6)

    mixed = BiologicalReactorState(
        s_s=w_a * state_a.s_s + w_b * state_b.s_s,
        s_o=w_a * state_a.s_o + w_b * state_b.s_o,
        s_no=w_a * state_a.s_no + w_b * state_b.s_no,
        s_nh=w_a * state_a.s_nh + w_b * state_b.s_nh,
        x_bh=w_a * state_a.x_bh + w_b * state_b.x_bh,
        x_ba=w_a * state_a.x_ba + w_b * state_b.x_ba,
    )
    return mixed, total_flow


def step(
    state: BiologicalReactorState,
    inlet: BiologicalReactorState,
    inlet_flow: jax.Array,
    kla: jax.Array,
    params: BiologicalReactorParams,
    dt: jax.Array,
) -> BiologicalReactorState:
    """Advance biological reactor by one time step using forward Euler.

    kla is the volumetric oxygen transfer coefficient (h⁻¹). Set kla=0 for
    anoxic operation. The Euler step is stable for typical BSM1 timesteps
    (dt ≤ 0.05 h) and standard kinetic constants.
    """
    # Hydraulic dilution rate (h⁻¹)
    D = inlet_flow / params.volume

    # Aerobic heterotrophic growth (COD removal with O₂)
    r_h_aero = (
        params.mu_h
        * (state.s_s / (params.k_s + state.s_s))
        * (state.s_o / (params.k_o_h + state.s_o))
        * state.x_bh
    )

    # Anoxic heterotrophic growth (denitrification with NO₃)
    r_h_anox = (
        params.mu_h
        * params.eta_g
        * (state.s_s / (params.k_s + state.s_s))
        * (params.k_o_h / (params.k_o_h + state.s_o))
        * (state.s_no / (params.k_no + state.s_no))
        * state.x_bh
    )

    # Autotrophic growth (nitrification)
    r_auto = (
        params.mu_a
        * (state.s_nh / (params.k_nh + state.s_nh))
        * (state.s_o / (params.k_o_a + state.s_o))
        * state.x_ba
    )

    # Decay
    r_decay_h = params.b_h * state.x_bh
    r_decay_a = params.b_a * state.x_ba

    # Mass balances (g m⁻³ h⁻¹)
    ds_s = (
        D * (inlet.s_s - state.s_s)
        - (r_h_aero + r_h_anox) / params.y_h
        + (1.0 - params.f_p) * (r_decay_h + r_decay_a)
    )
    ds_o = (
        D * (inlet.s_o - state.s_o)
        + kla * (params.s_o_sat - state.s_o)
        - (1.0 - params.y_h) / params.y_h * r_h_aero
        - (4.57 - params.y_a) / params.y_a * r_auto
    )
    ds_no = (
        D * (inlet.s_no - state.s_no)
        - (1.0 - params.y_h) / (2.86 * params.y_h) * r_h_anox
        + 1.0 / params.y_a * r_auto
    )
    ds_nh = (
        D * (inlet.s_nh - state.s_nh)
        - params.i_xb * (r_h_aero + r_h_anox)
        - (params.i_xb + 1.0 / params.y_a) * r_auto
    )
    dx_bh = D * (inlet.x_bh - state.x_bh) + (r_h_aero + r_h_anox) - r_decay_h
    dx_ba = D * (inlet.x_ba - state.x_ba) + r_auto - r_decay_a

    # Forward Euler with non-negativity enforcement
    return BiologicalReactorState(
        s_s=jnp.maximum(state.s_s + ds_s * dt, 0.0),
        s_o=jnp.clip(state.s_o + ds_o * dt, 0.0, params.s_o_sat),
        s_no=jnp.maximum(state.s_no + ds_no * dt, 0.0),
        s_nh=jnp.maximum(state.s_nh + ds_nh * dt, 0.0),
        x_bh=jnp.maximum(state.x_bh + dx_bh * dt, 0.0),
        x_ba=jnp.maximum(state.x_ba + dx_ba * dt, 0.0),
    )
