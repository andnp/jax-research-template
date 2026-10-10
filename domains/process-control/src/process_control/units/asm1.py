import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.binding import NamedAccess, Requirement
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.kinetics import ZERO_CELSIUS


@jax_dataclass
class ASM1Params:
    """Kinetic and stoichiometric parameters for the full ASM1 model.

    Default values follow BSM1 benchmark specification (Copp et al., 2002) at 15°C,
    converted from 1/day to 1/hour. References:
      Henze, M. et al. (1987). Activated Sludge Model No. 1. IAWPRC.
      Copp, J.B. (2002). The COST Simulation Benchmark. Office for Official
        Publications of the European Communities, Luxembourg.
    """

    # --- Heterotrophic kinetics ---
    mu_h: float | jax.Array = 0.16667  # max growth rate (h⁻¹; BSM1 at 15°C: 4.0 /day)
    k_s: float = 10.0  # half-saturation, biodegradable substrate (g COD/m³)
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
    b_a: float | jax.Array = 0.002083  # nitrifier decay rate (h⁻¹; 0.05 /day)
    y_a: float = 0.24  # yield (g COD / g N oxidised)

    # --- Hydrolysis ---
    k_h: float | jax.Array = 0.12500  # max hydrolysis rate (h⁻¹; 3.0 /day)
    k_x: float = 0.1  # half-saturation, X_S/X_BH ratio
    eta_h: float = 0.8  # anoxic correction factor for hydrolysis

    # --- Ammonification ---
    k_a: float | jax.Array = 0.002083  # rate constant (m³ gCOD⁻¹ h⁻¹; 0.05 /day)

    # --- Stoichiometric constants ---
    i_xb: float = 0.08  # N content of biomass (g N / g COD)
    i_xp: float = 0.06  # N content of X_P (g N / g COD)

    # --- Reference temperature for Arrhenius correction ---
    t_ref: float = 15.0  # °C (BSM1 default calibration temperature)


@dataclass(frozen=True)
class ArrheniusCoeffs:
    """Temperature coefficients for ASM1 rate constants.

    Each coefficient θ adjusts a rate via: rate(T) = rate(T_ref) × exp(θ × (T − T_ref)).
    θ = ln(rate_15°C / rate_10°C) / 5, from Henze et al. (1987) 10/20°C values as
    fit by Vanrolleghem, Batstone & Pons (2009), "Temperature effects in BSM1_LT
    and BSM2" (eqs. 1-6): https://wwtmodels.pubpub.org/pub/01quq2jb
    """

    theta_mu_h: float = 0.057536  # heterotrophic growth: ln(4.0/3.0)/5
    theta_mu_a: float = 0.102165  # autotrophic growth: ln(0.5/0.3)/5
    theta_b_h: float = 0.081093  # heterotrophic decay: ln(0.3/0.2)/5
    theta_b_a: float = 0.102165  # autotrophic decay: ln(0.05/0.03)/5
    theta_k_h: float = 0.036464  # hydrolysis: ln(3.0/2.5)/5
    theta_k_a: float = 0.044629  # ammonification: ln(0.05/0.04)/5


def bsm1_oxygen_saturation(temperature: jax.Array):
    """O₂ saturation (g/m³) at a temperature (°C) by the BSM1_LT/BSM2 convention.

    A Lide (2004) solubility correlation, scaled so it reproduces S_O,sat = 8 g/m³ at 15°C —
    the BSM1 reference value — rather than the true ~10.3 g/m³ physical solubility at that
    temperature. Vanrolleghem, Batstone & Pons (2009), "Temperature effects in BSM1_LT and
    BSM2", eqs. 12-13: https://wwtmodels.pubpub.org/pub/01quq2jb
    """
    t_k_star = (temperature + 273.15) / 100.0
    lide_a, lide_b, lide_c = -66.7354, 87.4755, 24.4526
    solubility = 56.12 * jnp.exp(lide_a + lide_b / t_k_star + lide_c * jnp.log(t_k_star))
    return 0.09521660204 * 8.0 * 6791.5 * solubility


def apply_arrhenius(
    params: ASM1Params,
    temperature: jax.Array,
    coeffs: ArrheniusCoeffs | None = None,
):
    """Return ASM1Params with rate constants corrected to the given temperature."""
    if coeffs is None:
        coeffs = ArrheniusCoeffs()
    dt = temperature - params.t_ref

    def _corr(base: float | jax.Array, theta: float):
        return base * jnp.exp(theta * dt)

    return dataclasses.replace(
        params,
        mu_h=_corr(params.mu_h, coeffs.theta_mu_h),
        b_h=_corr(params.b_h, coeffs.theta_b_h),
        mu_a=_corr(params.mu_a, coeffs.theta_mu_a),
        b_a=_corr(params.b_a, coeffs.theta_b_a),
        k_h=_corr(params.k_h, coeffs.theta_k_h),
        k_a=_corr(params.k_a, coeffs.theta_k_a),
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


ASM1_SCHEMA = ComponentSchema(
    names=("s_i", "s_s", "x_i", "x_s", "x_bh", "x_ba", "x_p", "s_o", "s_no", "s_nh", "s_nd", "x_nd", "s_alk"),
    particulate=(False, False, True, True, True, True, True, False, False, False, False, True, False),
    tss_factor=(0.0, 0.0, 0.75, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Alex et al. (2008), Table 6; alkalinity counts acid-neutralizing equivalents by ASM1 convention,
    # not bicarbonate concentration or total inorganic carbon.
    bases=(
        *(ComponentBasis("g/m3", Quantity.COD, "COD"),) * 7,
        ComponentBasis("g/m3", Quantity.SPECIES, "O2"),
        *(ComponentBasis("g/m3", Quantity.ELEMENT, "N"),) * 4,
        ComponentBasis("mol/m3", Quantity.EQUIVALENT, "HCO3-"),
    ),
)


def _asm1_requirements(*names: str) -> tuple[Requirement, ...]:
    return tuple(Requirement(name, ASM1_SCHEMA.bases[ASM1_SCHEMA.index(name)]) for name in names)


# s_i and x_i take part in no reaction; x_p and s_alk are produced but never limit a rate.
ASM1_ACCESS = NamedAccess(
    reads=_asm1_requirements("s_s", "x_s", "x_bh", "x_ba", "s_o", "s_no", "s_nh", "s_nd", "x_nd"),
    writes=_asm1_requirements("s_s", "x_s", "x_bh", "x_ba", "x_p", "s_o", "s_no", "s_nh", "s_nd", "x_nd", "s_alk"),
)

def compute_tss(state: ASM1State) -> jax.Array:
    """Total suspended solids (g TSS/m³). Uses BSM1 conversion factors."""
    return 0.75 * (state.x_i + state.x_s + state.x_bh + state.x_ba + state.x_p)


def compute_cod(state: ASM1State) -> jax.Array:
    """Chemical oxygen demand (g COD/m³)."""
    return state.s_i + state.s_s + state.x_i + state.x_s + state.x_bh + state.x_ba + state.x_p


def compute_bod5(state: ASM1State, params: ASM1Params) -> jax.Array:
    """Biochemical oxygen demand over 5 days (g O₂/m³).

    BSM1 effluent quality definition (Alex et al., 2008, eq. in section 6):
    https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
    """
    return 0.25 * (state.s_s + state.x_s + (1.0 - params.f_p) * (state.x_bh + state.x_ba))


def compute_tkn(state: ASM1State, params: ASM1Params) -> jax.Array:
    """Total Kjeldahl nitrogen (g N/m³): BSM1 S_Nkj definition (Alex et al., 2008)."""
    return (
        state.s_nh
        + state.s_nd
        + state.x_nd
        + params.i_xb * (state.x_bh + state.x_ba)
        + params.i_xp * (state.x_p + state.x_i)
    )


def compute_tn(state: ASM1State, params: ASM1Params) -> jax.Array:
    """Total nitrogen (g N/m³): TKN + nitrate/nitrite (BSM1 N_tot)."""
    return compute_tkn(state, params) + state.s_no


def asm1_reaction_rates(reads: jax.Array, params: ASM1Params):
    """ASM1 biological reaction rates per hour of the `ASM1_ACCESS` writes, from its reads along the last axis."""
    s_s, x_s, x_bh, x_ba, s_o, s_no, s_nh, s_nd, x_nd = jnp.unstack(reads, axis=-1)

    def monod(s: jax.Array, k: float) -> jax.Array:
        return s / (k + s)

    def inhibition(s: jax.Array, k: float) -> jax.Array:
        return k / (k + s)

    rho_1 = params.mu_h * monod(s_s, params.k_s) * monod(s_o, params.k_o_h) * x_bh
    rho_2 = params.mu_h * monod(s_s, params.k_s) * inhibition(s_o, params.k_o_h) * monod(s_no, params.k_no) * params.eta_g * x_bh
    rho_3 = params.mu_a * monod(s_nh, params.k_nh) * monod(s_o, params.k_o_a) * x_ba
    rho_4 = params.b_h * x_bh
    rho_5 = params.b_a * x_ba
    rho_6 = params.k_a * s_nd * x_bh
    x_ratio = x_s / (x_bh + 1e-10)
    rho_7 = params.k_h * x_ratio / (params.k_x + x_ratio) * (monod(s_o, params.k_o_h) + params.eta_h * inhibition(s_o, params.k_o_h) * monod(s_no, params.k_no)) * x_bh
    rho_8 = rho_7 * (x_nd / (x_s + 1e-10))

    alk_anox = (1.0 - params.y_h) / (14.0 * 2.86 * params.y_h) - params.i_xb / 14.0
    alk_aer = params.i_xb / 14.0 + 1.0 / (7.0 * params.y_a)
    return jnp.stack(
        [
            -rho_1 / params.y_h - rho_2 / params.y_h + rho_7,
            (1.0 - params.f_p) * rho_4 + (1.0 - params.f_p) * rho_5 - rho_7,
            rho_1 + rho_2 - rho_4,
            rho_3 - rho_5,
            params.f_p * rho_4 + params.f_p * rho_5,
            -(1.0 - params.y_h) / params.y_h * rho_1 - (4.57 - params.y_a) / params.y_a * rho_3,
            -(1.0 - params.y_h) / (2.86 * params.y_h) * rho_2 + rho_3 / params.y_a,
            -params.i_xb * rho_1 - params.i_xb * rho_2 - (params.i_xb + 1.0 / params.y_a) * rho_3 + rho_6,
            -rho_6 + rho_8,
            (params.i_xb - params.f_p * params.i_xp) * rho_4 + (params.i_xb - params.f_p * params.i_xp) * rho_5 - rho_8,
            -params.i_xb / 14.0 * rho_1 + alk_anox * rho_2 - alk_aer * rho_3 + rho_6 / 14.0,
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class ASM1Kinetics:
    """ASM1 kinetics with rate constants Arrhenius-corrected from their `t_ref` (°C) values on every call."""

    arrhenius: ArrheniusCoeffs = ArrheniusCoeffs()

    @property
    def access(self) -> NamedAccess:
        return ASM1_ACCESS

    def rates(self, reads: jax.Array, temperature_k: jax.Array, params: ASM1Params) -> jax.Array:
        return asm1_reaction_rates(reads, apply_arrhenius(params, temperature_k - ZERO_CELSIUS, self.arrhenius))
