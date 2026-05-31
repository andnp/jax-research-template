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


@dataclass(frozen=True)
class ScrubberSumpParams:
    """Sump geometry and reduced-order chemistry constants."""

    # ── Geometry ──────────────────────────────────────────────────
    volume_min: float = 1.0  # m³  – low-level alarm / clamp
    volume_max: float = 10.0  # m³  – overflow point
    nominal_volume: float = 5.0  # m³  – design operating level

    # ── Oxidant (bleach) kinetics ─────────────────────────────────
    bleach_yield: float = 1.0  # mg-oxidant / mL-bleach-pumped
    oxidant_decay: float = 0.02  # h⁻¹  – natural decomposition (NaOCl → NaCl + O₂)
    oxidant_sulfide_rate: float = 2.0  # mg-oxidant consumed / mg-sulfide oxidised

    # ── Alkalinity (caustic) kinetics ─────────────────────────────
    caustic_yield: float = 0.5  # meq/L per mL-caustic-pumped per m³
    alkalinity_consumption: float = 0.1  # meq consumed per mg-sulfide absorbed

    # ── Sulfide kinetics ──────────────────────────────────────────
    oxidation_rate: float = 5.0  # mg-sulfide/h per (mg/L oxidant) — first-order-ish
    stripping_rate: float = 0.5  # h⁻¹  – dissolved sulfide re-stripped to gas

    # ── Temperature ───────────────────────────────────────────────
    ambient_temp: float = 25.0  # °C
    temp_exchange_rate: float = 0.1  # h⁻¹  – heat loss to ambient
    reaction_heat: float = 0.02  # °C per mg-sulfide oxidised (exothermic)

    # ── Temperature effect on kinetics ────────────────────────────
    temp_reference: float = 25.0  # °C  – reference for Arrhenius-like scaling
    temp_coefficient: float = 1.05  # θ  – rate multiplier per °C above ref


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
        oxidant: float = 5.0,
        alkalinity: float = 3.0,
        volume: float = 5.0,
        sulfide: float = 0.5,
        temperature: float = 25.0,
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

    # ── Alkalinity balance ────────────────────────────────────────
    # caustic_flow is mL/min → mL/h: × 60
    caustic_addition = params.caustic_yield * inputs.caustic_flow * 60.0 / vol
    alk_consumed = params.alkalinity_consumption * oxidation
    dilution_alk = makeup_m3_h / vol * state.alkalinity
    d_alkalinity = caustic_addition - alk_consumed - dilution_alk

    # ── Sulfide balance ───────────────────────────────────────────
    # sulfide_load is mg/min → mg/L/h: × 60 / (vol_L)
    sulfide_addition = inputs.sulfide_load * 60.0 / (vol * 1000.0)
    sulfide_oxidised = oxidation
    sulfide_stripped = params.stripping_rate * state.sulfide
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


def step(
    state: ScrubberSumpState,
    inputs: SumpInputs,
    params: ScrubberSumpParams,
    dt: jax.Array,
) -> ScrubberSumpState:
    """Advance sump state by one timestep using RK4 integration.

    Args:
        state: current sump state
        inputs: external flows (held constant over the step)
        params: sump parameters
        dt: timestep in hours
    """
    new_state = rk4_step(_derivatives, state, dt, inputs, params)

    # Enforce physical bounds (non-negative concentrations, volume limits)
    return ScrubberSumpState(
        oxidant=jnp.maximum(new_state.oxidant, 0.0),
        alkalinity=jnp.maximum(new_state.alkalinity, 0.0),
        volume=jnp.clip(new_state.volume, params.volume_min, params.volume_max * 1.5),
        sulfide=jnp.maximum(new_state.sulfide, 0.0),
        temperature=jnp.clip(new_state.temperature, 5.0, 60.0),
    )


# ── Derived quantities ────────────────────────────────────────────────


def compute_ph(state: ScrubberSumpState) -> jax.Array:
    """Approximate sump pH from alkalinity state.

    Simple sigmoid mapping: low alkalinity → acidic, high → basic.
    Real pH depends on full carbonate/hydroxide speciation, but this
    captures the monotonic caustic→pH relationship the controller needs.
    """
    return jnp.array(7.0) + jnp.array(3.0) * jnp.tanh((state.alkalinity - 2.0) / 3.0)


def compute_orp(state: ScrubberSumpState) -> jax.Array:
    """Approximate ORP (mV) from oxidant and sulfide state.

    ORP rises with oxidant availability and drops with dissolved sulfide.
    Typical scrubber ORP range: 200–800 mV.
    """
    # Nernst-like: ORP ∝ log(oxidant / sulfide)
    ratio = (state.oxidant + 0.01) / (state.sulfide + 0.01)
    return jnp.array(450.0) + jnp.array(100.0) * jnp.log10(ratio)
