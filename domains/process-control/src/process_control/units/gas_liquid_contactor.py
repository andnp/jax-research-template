"""Reduced-order gas-liquid contactor for H₂S scrubbing.

Maps inlet gas conditions and liquid-phase chemistry to:
  - H₂S removal efficiency (0–1)
  - outlet H₂S concentration
  - sulfide load transferred to the scrubbing liquid

The contactor is stateless — it's a transfer function, not a dynamic
system.  All dynamics live in the sump chemistry model.

Removal efficiency depends on:
  - oxidant availability (bleach) — directly enables sulfide oxidation
  - alkalinity (caustic) — affects gas-liquid equilibrium and absorption
  - gas flow rate — higher flow reduces contact time, lowers efficiency
  - liquid recirculation rate — higher recirc improves mass transfer
  - gas temperature — H₂S solubility falls as temperature rises (Henry's
    law), so absorption efficiency does too
  - liquid pH — sets the H₂S/HS⁻ speciation equilibrium, which in turn
    sets the residual H₂S vapor pressure the liquid can still absorb
    against; not the same effect as alkalinity's Monod term above
  - dissolved sulfide already in the liquid — absorption is driven by
    the gap between the liquid's equilibrium and actual H₂S loading
    (two-film theory), so a liquid that is already loaded up absorbs
    less regardless of oxidant/alkalinity/gas-flow conditions

References for the mechanisms above (this reduced-order model is not
itself derived from any of these, but its functional shape is chosen to
be consistent with them):
  - NaOH/H₂S scrubbing reactions and the pH-dependent NaHS/Na₂S product
    split: Trimeric, "Caustic Scrubber Designs for H₂S Removal from
    Refinery Gas Streams" (AFPM AM-14-48, 2014),
    https://trimeric.com/assets/caustic-scrubber-designs-for-h2s-removal-from-refinery-gas-streams---afpm-2014.pdf
  - H₂S Henry's law constant and its temperature dependence: NIST
    WebBook, https://webbook.nist.gov/cgi/cbook.cgi?ID=C7783064
  - pH/alkalinity/temperature/L-G dependence of packed-column H₂S
    absorption: Azizi et al., https://pubmed.ncbi.nlm.nih.gov/25244138/
  - Two-film theory of gas absorption (Lewis & Whitman, 1924): rate
    N = kL·a·(C* - C_L), where C* is the liquid concentration in
    equilibrium with the gas and C_L is the liquid's actual dissolved
    load; this model's `loading_half_sat` term is a reduced-order proxy
    for that driving-force gap, using the sump's `sulfide` state as a
    stand-in for C_L rather than a literal Henry's-law concentration.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@jax_dataclass
class ContactorParams:
    """Gas-liquid contactor parameters."""

    # ── Base transfer ─────────────────────────────────────────────
    max_efficiency: float | jax.Array = 0.99  # theoretical max removal at perfect conditions
    base_efficiency: float | jax.Array = 0.90  # removal at nominal conditions

    # ── Oxidant effect ────────────────────────────────────────────
    oxidant_half_sat: float | jax.Array = 3.0  # mg/L – half-saturation for oxidant effect
    oxidant_weight: float | jax.Array = 0.5  # relative importance (0–1)

    # ── Alkalinity effect ─────────────────────────────────────────
    alkalinity_half_sat: float | jax.Array = 2.0  # meq/L – half-saturation
    alkalinity_weight: float | jax.Array = 0.3  # relative importance (0–1)

    # ── Gas flow effect ───────────────────────────────────────────
    nominal_gas_flow: float | jax.Array = 500.0  # m³/h – design gas flow
    gas_flow_exponent: float | jax.Array = -0.3  # efficiency ∝ (Q/Q_nom)^exp (negative = higher flow → lower eff)

    # ── Recirculation effect ──────────────────────────────────────
    nominal_recirc: float | jax.Array = 50.0  # m³/h – design recirculation rate
    recirc_weight: float | jax.Array = 0.2  # relative importance (0–1)
    recirc_half_sat: float | jax.Array = 25.0  # m³/h – half-saturation

    # ── pH speciation effect ───────────────────────────────────────
    # Henderson-Hasselbalch factor for the H2S/HS- equilibrium; see
    # module docstring (Trimeric, Azizi et al.). 7.0 is H2S's consensus
    # aqueous pKa1 (NIST/EPA IRIS literature range 6.97-7.06 at 25°C).
    h2s_pka1: float | jax.Array = 7.0

    # ── Temperature effect ────────────────────────────────────────
    # Van't Hoff/Arrhenius-style decay, mirroring ScrubberSumpParams'
    # reaction-rate temperature term but inverted: absorption efficiency
    # falls (not rises) as temperature climbs above the reference point.
    temp_reference: float | jax.Array = 15.0  # °C – reference gas temperature
    temp_coefficient: float | jax.Array = 0.98  # efficiency multiplier per °C above reference

    # ── Liquid loading effect ──────────────────────────────────────
    # Reduced-order proxy for two-film theory's driving-force gap (see
    # module docstring): absorption throttles as the sump's dissolved
    # sulfide state approaches this half-saturation level, standing in
    # for the liquid's H2S load approaching its gas-equilibrium value.
    loading_half_sat: float | jax.Array = 10.0  # mg/L


@dataclass(frozen=True)
class GasInlet:
    """Incoming gas stream conditions."""

    gas_flow: jax.Array  # m³/h
    h2s_ppm: jax.Array  # ppmv H₂S in inlet gas
    temperature: jax.Array  # °C (affects Henry's law, not modeled in L1)


@dataclass(frozen=True)
class ContactorResult:
    """Contactor outputs for one timestep."""

    removal_efficiency: jax.Array  # 0–1
    outlet_h2s_ppm: jax.Array  # ppmv
    sulfide_load: jax.Array  # mg/min transferred to liquid


def compute_removal(
    gas_inlet: GasInlet,
    oxidant: jax.Array,
    alkalinity: jax.Array,
    recirc_flow: jax.Array,
    ph: jax.Array,
    sulfide: jax.Array,
    params: ContactorParams,
) -> ContactorResult:
    """Compute H₂S removal for current conditions.

    Args:
        gas_inlet: incoming gas conditions
        oxidant: current sump oxidant level (mg/L)
        alkalinity: current sump alkalinity (meq/L)
        recirc_flow: liquid recirculation rate (m³/h)
        ph: current sump liquid pH
        sulfide: current sump dissolved sulfide level (mg/L)
        params: contactor parameters
    """
    # ── Oxidant factor: Monod saturation ──────────────────────────
    f_oxidant = oxidant / (oxidant + params.oxidant_half_sat)

    # ── Alkalinity factor: Monod saturation ───────────────────────
    f_alkalinity = alkalinity / (alkalinity + params.alkalinity_half_sat)

    # ── pH speciation factor: Henderson-Hasselbalch ───────────────
    # Fraction of dissolved sulfide present as HS- rather than
    # un-ionized H2S(aq); un-ionized H2S sets the equilibrium vapor
    # pressure the liquid absorbs against, so removal is throttled as
    # pH falls toward pKa1 even with ample oxidant/alkalinity present.
    f_ph = 1.0 / (1.0 + jnp.power(10.0, params.h2s_pka1 - ph))

    # ── Liquid loading factor: two-film driving-force proxy ───────
    # Absorption slows as the liquid's dissolved sulfide load approaches
    # the equilibrium it would reach with the gas phase; see module
    # docstring (Lewis & Whitman two-film theory).
    f_loading = params.loading_half_sat / (params.loading_half_sat + sulfide)

    # ── Gas flow factor: power law ────────────────────────────────
    flow_ratio = gas_inlet.gas_flow / params.nominal_gas_flow
    f_gas = jnp.power(jnp.maximum(flow_ratio, 0.01), params.gas_flow_exponent)

    # ── Recirculation factor: Monod saturation ────────────────────
    f_recirc = recirc_flow / (recirc_flow + params.recirc_half_sat)

    # ── Temperature factor: Van't Hoff/Arrhenius-style decay ──────
    # H₂S gas-liquid solubility falls with rising temperature (Henry's
    # law), so absorption efficiency does too; see module docstring.
    f_temperature = jnp.power(params.temp_coefficient, gas_inlet.temperature - params.temp_reference)

    # ── Weighted combination ──────────────────────────────────────
    # Chemistry factors (oxidant + alkalinity) and hydraulic factors (gas, recirc)
    chem_factor = f_ph * f_loading * (params.oxidant_weight * f_oxidant + params.alkalinity_weight * f_alkalinity) / (params.oxidant_weight + params.alkalinity_weight)

    hydraulic_factor = f_gas * ((1.0 - params.recirc_weight) + params.recirc_weight * f_recirc)

    efficiency = params.base_efficiency * chem_factor * hydraulic_factor * f_temperature
    efficiency = jnp.clip(efficiency, 0.0, params.max_efficiency)

    # ── Outlet H₂S ───────────────────────────────────────────────
    outlet_h2s = gas_inlet.h2s_ppm * (1.0 - efficiency)

    # ── Sulfide mass transfer to liquid ───────────────────────────
    # H₂S removed from gas → dissolved sulfide in liquid
    # Approximate: ppmv × gas_flow → mass rate
    # At STP: 1 ppmv H₂S in 1 m³/h ≈ 1.43 mg/h of H₂S
    h2s_density_factor = 1.43 / 60.0  # mg/min per (ppmv × m³/h)
    removed_h2s_ppm = gas_inlet.h2s_ppm * efficiency
    sulfide_load = removed_h2s_ppm * gas_inlet.gas_flow * h2s_density_factor

    return ContactorResult(
        removal_efficiency=efficiency,
        outlet_h2s_ppm=outlet_h2s,
        sulfide_load=sulfide_load,
    )
