"""Reduced-order recirculating scrubber sump chemistry model.

Tracks five liquid-phase state variables that give the sump "memory":

  1. oxidant  – available bleach oxidant (mg/L equivalent)
  2. alkalinity – caustic-driven alkalinity (meq/L equivalent)
  3. volume  – liquid inventory (m³)
  4. sulfide – dissolved sulfide from H₂S absorption (mg/L)
  5. temperature – liquid temperature (°C) (slow drift, not actively controlled)

Chemistry is deliberately reduced-order: we model *rates of change* from
dosing, consumption, dilution, and overflow — not full speciation.  This is
enough to reproduce the real couplings that make scrubber control hard:

  - bleach and caustic are *not* interchangeable
  - oxidant depletes under high sulfur load
  - makeup dilutes everything but refreshes volume
  - temperature affects reaction rates

Uses RK4 integration for numerical stability.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.integration import rk4_step

# ── Parameters ────────────────────────────────────────────────────────


@jax_dataclass
class ScrubberSumpParams:
    """Sump geometry and reduced-order chemistry constants."""

    # ── Geometry ──────────────────────────────────────────────────
    volume_min: float | jax.Array = 1.0  # m³  – low-level alarm / clamp
    volume_max: float | jax.Array = 10.0  # m³  – overflow point
    nominal_volume: float | jax.Array = 5.0  # m³  – design operating level

    # ── Oxidant (bleach) kinetics ─────────────────────────────────
    # `oxidant` and `sulfide` are both tracked on a mg-NaOCl-equivalent /
    # mg-H2S-equivalent basis (matching the contactor's sulfide_load, which
    # is mass-transferred H2S, not elemental S) — no extra 32/34 conversion
    # is needed between them.
    bleach_yield: float | jax.Array = 1.0  # mg-oxidant (as NaOCl) / mL-bleach-pumped
    oxidant_decay: float | jax.Array = 0.02  # h⁻¹  – natural decomposition (NaOCl → NaCl + O₂)
    # At scrubber pH (~9-10) hypochlorite drives sulfide oxidation mostly to
    # sulfate rather than elemental S (HS- + 4OCl- -> SO4^2- + 4Cl- + H+ vs.
    # HS- + OCl- -> S + Cl- + OH- at lower pH/limited oxidant); the sulfate
    # route needs ~8.7 mg NaOCl per mg H2S (4 x 74.44/34.08), not ~2.2 (Choppin
    # & Faulkenberry, "The Oxidation of Aqueous Sulfide Solutions by
    # Hypochlorites," J. Am. Chem. Soc. 59(11):2203-2207, 1937,
    # https://doi.org/10.1021/ja01290a033):
    oxidant_sulfide_rate: float | jax.Array = 8.7  # mg-oxidant consumed / mg-sulfide oxidised

    # ── Alkalinity (caustic) kinetics ─────────────────────────────
    # meq/L per mL-caustic-pumped per m3 == Normality(eq/L) / 1000 (the m3->L
    # and mL->L factors of 1000 cancel — see _derivatives' caustic_addition).
    # 50% NaOH: density 1.515 g/mL (https://www.laboratorynotes.com/molarity-of-50-percent-sodium-hydroxide-naoh/),
    # NaOH MW 40.00 g/mol, 1 eq/mol -> N = 1.515*1000*0.50/40 ~= 18.9 eq/L.
    caustic_yield: float | jax.Array = 0.019  # meq/L per mL-caustic-pumped per m³ (50% NaOH)
    # H2S + OH- -> HS- + H2O consumes 1 eq per mol H2S absorbed (MW 34.08
    # g/mol) -> 1/34.08 = 0.0293 meq/mg; the further acid released by
    # oxidation to sulfate is charged separately via
    # oxidation_alkalinity_consumption below, not here.
    alkalinity_consumption: float | jax.Array = 0.0293  # meq consumed per mg-sulfide absorbed
    # HS- + 4OCl- -> SO4^2- + 4Cl- + H+ releases 1 eq of acid per mol S
    # oxidized to sulfate (same 1/34.08 mol-to-mass factor as absorption
    # above), neutralized by alkalinity rather than by more caustic dosing.
    oxidation_alkalinity_consumption: float | jax.Array = 0.0293  # meq consumed per mg-sulfide oxidised

    # ── Sulfide kinetics ──────────────────────────────────────────
    oxidation_rate: float | jax.Array = 5.0  # mg-sulfide/h per (mg/L oxidant) — first-order-ish
    stripping_rate: float | jax.Array = 0.5  # h⁻¹  – dissolved H2S(aq) re-stripped to gas
    # Only un-ionized H2S(aq), not HS-, has a gas-phase vapor pressure to
    # strip against; see contactor's f_ph (same Henderson-Hasselbalch split).
    h2s_pka1: float | jax.Array = 7.0  # aqueous pKa1, NIST/EPA IRIS range 6.97-7.06 at 25°C
    ph_amplitude: float | jax.Array = 3.0  # must match H2SScrubberConfig.sump_ph_amplitude

    # ── Temperature ───────────────────────────────────────────────
    ambient_temp: float | jax.Array = 25.0  # °C
    temp_exchange_rate: float | jax.Array = 0.1  # h⁻¹  – heat loss to ambient
    # H2S(g) + 4ClO-(aq) -> SO4^2-(aq) + 4Cl-(aq) + 2H+(aq), Hess's law from
    # standard formation enthalpies (H2S(g) -20.6, SO4^2-(aq) -909.34,
    # Cl-(aq) -167.08 kJ/mol: https://www.mrbigler.com/misc/energy-of-formation.PDF;
    # ClO-(aq) -107.1 kJ/mol, commonly tabulated general-chemistry value,
    # cross-checked but not from a single primary source) gives
    # ~-1129 kJ/mol H2S, i.e. 33.1 J/mg; divided by 1 L water's heat capacity
    # (4.184 J/g/°C * 1000 g/L = 4184 J/(L*°C)) that is ~0.0079 °C per mg/L
    # oxidised — using H2S(g) rather than H2S(aq) formation enthalpy (not
    # tabulated in the source above) makes this an approximate, not exact,
    # figure, but it corrects the prior value's ~2.5x overshoot.
    reaction_heat: float | jax.Array = 0.008  # °C per mg-sulfide oxidised (exothermic)

    # ── Temperature effect on kinetics ────────────────────────────
    temp_reference: float | jax.Array = 25.0  # °C  – reference for Arrhenius-like scaling
    temp_coefficient: float | jax.Array = 1.05  # θ  – rate multiplier per °C above ref

    # ── ORP pH dependence ─────────────────────────────────────────
    # Nernstian slope for the hypochlorite ORP/pH term (see compute_orp);
    # 59.2 mV/pH is the standard 25°C Nernst factor (RT ln(10) / F), but a
    # real platinum ORP electrode's response is a mixed potential, not a
    # clean Nernstian one (Pt oxide formation passivates the electrode),
    # so a fitted slope well above 59.2 is expected, not a red flag — see
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC6111276/
    # orp_ph_reference anchors the existing 450 mV baseline at a typical
    # plant operating pH so the term is a correction, not a new offset.
    orp_ph_slope: float | jax.Array = 59.2  # mV per pH unit
    orp_ph_reference: float | jax.Array = 9.8  # pH


# ── State ─────────────────────────────────────────────────────────────


@jax_dataclass
class ScrubberSumpState:
    oxidant: jax.Array  # mg/L  – available bleach oxidant
    alkalinity: jax.Array  # meq/L – caustic-driven alkalinity
    volume: jax.Array  # m³    – liquid level
    sulfide: jax.Array  # mg/L  – dissolved sulfide
    temperature: jax.Array  # °C

    @staticmethod
    def create(
        oxidant: float | jax.Array = 5.0,
        alkalinity: float | jax.Array = 3.0,
        volume: float | jax.Array = 5.0,
        sulfide: float | jax.Array = 0.5,
        temperature: float | jax.Array = 25.0,
    ) -> "ScrubberSumpState":
        return ScrubberSumpState(
            oxidant=jnp.array(oxidant),
            alkalinity=jnp.array(alkalinity),
            volume=jnp.array(volume),
            sulfide=jnp.array(sulfide),
            temperature=jnp.array(temperature),
        )


# ── Inputs ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SumpInputs:
    """External flows entering the sump each timestep."""

    bleach_flow: jax.Array  # mL/min – bleach pump output
    caustic_flow: jax.Array  # mL/min – caustic pump output
    makeup_flow: jax.Array  # L/min  – makeup water flow
    sulfide_load: jax.Array  # mg/min – dissolved sulfide entering from contactor


# ── Reset ─────────────────────────────────────────────────────────────


def reset(rng_key: jax.Array, params: ScrubberSumpParams) -> ScrubberSumpState:
    return ScrubberSumpState.create(
        volume=params.nominal_volume,
    )


# ── Derivatives (for RK4) ────────────────────────────────────────────


def _derivatives(
    state: ScrubberSumpState,
    inputs: SumpInputs,
    params: ScrubberSumpParams,
) -> ScrubberSumpState:
    """Compute time derivatives of sump state (units per hour)."""
    vol = jnp.maximum(state.volume, params.volume_min)

    # Temperature-dependent rate scaling (Arrhenius-like)
    temp_factor = params.temp_coefficient ** (state.temperature - params.temp_reference)

    # ── Sulfide oxidation (oxidant + sulfide → products) ──────────
    # Rate limited by both oxidant and sulfide availability
    oxidation = (
        params.oxidation_rate * temp_factor * state.oxidant * state.sulfide / (state.sulfide + 1.0)  # Monod-like saturation
    )

    # ── Volume balance (L and m³ conversions) ─────────────────────
    # makeup_flow is L/min → m³/h: × 60/1000
    makeup_m3_h = inputs.makeup_flow * 60.0 / 1000.0
    # Overflow: soft clamp when above max volume
    overflow_rate = jnp.maximum(vol - params.volume_max, 0.0) * 10.0  # fast drain
    d_volume = makeup_m3_h - overflow_rate

    # ── Oxidant balance ───────────────────────────────────────────
    # bleach_flow is mL/min → mL/h: × 60
    bleach_addition = params.bleach_yield * inputs.bleach_flow * 60.0 / (vol * 1000.0)  # mg/L/h
    oxidant_consumed = params.oxidant_sulfide_rate * oxidation  # mg/L/h
    natural_decay = params.oxidant_decay * state.oxidant * temp_factor
    dilution_oxidant = makeup_m3_h / vol * state.oxidant
    d_oxidant = bleach_addition - oxidant_consumed - natural_decay - dilution_oxidant

    # ── Sulfide absorption (needed by the alkalinity balance below) ──
    # sulfide_load is mg/min → mg/L/h: × 60 / (vol_L)
    sulfide_addition = inputs.sulfide_load * 60.0 / (vol * 1000.0)

    # ── Alkalinity balance ────────────────────────────────────────
    # caustic_flow is mL/min → mL/h: × 60
    caustic_addition = params.caustic_yield * inputs.caustic_flow * 60.0 / vol
    # H2S(g) + OH- -> HS- + H2O: absorbing H2S directly neutralizes caustic,
    # independent of whether bleach later oxidises the dissolved sulfide.
    alk_consumed = params.alkalinity_consumption * sulfide_addition
    # The oxidation step to sulfate releases its own acid (see
    # oxidation_alkalinity_consumption above), a second, independent draw on
    # alkalinity keyed to how much sulfide is actually oxidised, not absorbed.
    alk_consumed_by_oxidation = params.oxidation_alkalinity_consumption * oxidation
    dilution_alk = makeup_m3_h / vol * state.alkalinity
    d_alkalinity = caustic_addition - alk_consumed - alk_consumed_by_oxidation - dilution_alk

    # ── Sulfide balance ───────────────────────────────────────────
    # Only un-ionized H2S(aq) has a vapor pressure to strip against; HS-
    # stays in solution (Henderson-Hasselbalch, same form as the contactor's
    # f_ph — see module docstring there).
    ph = compute_ph(state, params.ph_amplitude)
    h2s_aq_fraction = 1.0 / (1.0 + jnp.power(10.0, ph - params.h2s_pka1))
    sulfide_oxidised = oxidation
    sulfide_stripped = params.stripping_rate * state.sulfide * h2s_aq_fraction
    dilution_sulfide = makeup_m3_h / vol * state.sulfide
    d_sulfide = sulfide_addition - sulfide_oxidised - sulfide_stripped - dilution_sulfide

    # ── Temperature balance ───────────────────────────────────────
    heat_loss = params.temp_exchange_rate * (state.temperature - params.ambient_temp)
    heat_gain = params.reaction_heat * oxidation
    d_temperature = heat_gain - heat_loss

    return ScrubberSumpState(
        oxidant=d_oxidant,
        alkalinity=d_alkalinity,
        volume=d_volume,
        sulfide=d_sulfide,
        temperature=d_temperature,
    )


# ── Step ──────────────────────────────────────────────────────────────


def _clamp(state: ScrubberSumpState, params: ScrubberSumpParams) -> ScrubberSumpState:
    """Enforce physical bounds (non-negative concentrations, volume limits)."""
    return ScrubberSumpState(
        oxidant=jnp.maximum(state.oxidant, 0.0),
        alkalinity=jnp.maximum(state.alkalinity, 0.0),
        volume=jnp.clip(state.volume, params.volume_min, params.volume_max * 1.5),
        sulfide=jnp.maximum(state.sulfide, 0.0),
        temperature=jnp.clip(state.temperature, 5.0, 60.0),
    )


def step(
    state: ScrubberSumpState,
    inputs: SumpInputs,
    params: ScrubberSumpParams,
    dt: jax.Array,
    substeps: int = 10,
) -> tuple[ScrubberSumpState, jax.Array]:
    """Advance sump state by one timestep using RK4 integration.

    Splits `dt` into `substeps` RK4 sub-steps, clamping physical bounds after
    each one. At extreme (e.g. miscalibrated) reaction-rate parameters, a
    single explicit RK4 step over the full external timestep can fall outside
    the integrator's stability region and diverge to NaN within one step;
    finer sub-stepping with intermediate clamping keeps the integration
    stable without changing the external step interval or ever handing a
    negative concentration to a term with a pole there (e.g. the contactor's
    Monod-saturation factors).

    Args:
        state: current sump state
        inputs: external flows (held constant over the step)
        params: sump parameters
        dt: timestep in hours
        substeps: number of RK4 sub-steps per call
    """
    sub_dt = dt / substeps

    new_state = state
    total_overflow = jnp.array(0.0)
    for _ in range(substeps):
        raw_state = rk4_step(_derivatives, new_state, sub_dt, inputs, params)
        total_overflow += jnp.maximum(raw_state.volume - params.volume_max * 1.5, 0.0)
        new_state = _clamp(raw_state, params)
    return new_state, total_overflow


# ── Derived quantities ────────────────────────────────────────────────


def compute_ph(state: ScrubberSumpState, amplitude: float | jax.Array = 3.0) -> jax.Array:
    """Approximate sump pH from alkalinity state.

    Simple sigmoid mapping: low alkalinity → acidic, high → basic.
    Real pH depends on full carbonate/hydroxide speciation, but this
    captures the monotonic caustic→pH relationship the controller needs.
    `amplitude` sets the ceiling (7 + amplitude) and floor (7 - amplitude);
    it must exceed the real plant's observed pH range or the model
    saturates and stops responding to further alkalinity changes there.
    """
    return jnp.array(7.0) + jnp.array(amplitude) * jnp.tanh((state.alkalinity - 2.0) / 3.0)


def compute_orp(
    state: ScrubberSumpState,
    ph: jax.Array,
    params: ScrubberSumpParams,
) -> jax.Array:
    """Approximate ORP (mV) from oxidant, sulfide, and pH.

    ORP rises with oxidant availability and drops with dissolved sulfide.
    Typical scrubber ORP range: 200–800 mV.

    For a real hypochlorite system the Nernst relation also carries a pH
    term (OCl- + H2O + 2e- -> Cl- + 2OH-, so E = E0 - (RT/2F) ln([Cl-][OH-]^2/[ClO-]),
    i.e. ORP falls as pH rises for fixed oxidant) — see Emerson, "ORP Measurement Fundamentals",
    https://www.emerson.co.jp/is/content/emerson/en/measurement-instrumentation/technical/products/unaudited/documents/dl-rmt-00800-0100-4044.pdf
    """
    # Nernst-like: ORP ∝ log(oxidant / sulfide)
    ratio = (state.oxidant + 0.01) / (state.sulfide + 0.01)
    ph_correction = params.orp_ph_slope * (ph - params.orp_ph_reference)
    return jnp.array(450.0) + jnp.array(100.0) * jnp.log10(ratio) - ph_correction
