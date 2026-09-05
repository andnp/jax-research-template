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
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ContactorParams:
    """Gas-liquid contactor parameters."""

    # ── Base transfer ─────────────────────────────────────────────
    max_efficiency: float = 0.99  # theoretical max removal at perfect conditions
    base_efficiency: float = 0.90  # removal at nominal conditions

    # ── Oxidant effect ────────────────────────────────────────────
    oxidant_half_sat: float = 3.0  # mg/L – half-saturation for oxidant effect
    oxidant_weight: float = 0.5  # relative importance (0–1)

    # ── Alkalinity effect ─────────────────────────────────────────
    alkalinity_half_sat: float = 2.0  # meq/L – half-saturation
    alkalinity_weight: float = 0.3  # relative importance (0–1)

    # ── Gas flow effect ───────────────────────────────────────────
    nominal_gas_flow: float = 500.0  # m³/h – design gas flow
    gas_flow_exponent: float = -0.3  # efficiency ∝ (Q/Q_nom)^exp (negative = higher flow → lower eff)

    # ── Recirculation effect ──────────────────────────────────────
    nominal_recirc: float = 50.0  # m³/h – design recirculation rate
    recirc_weight: float = 0.2  # relative importance (0–1)
    recirc_half_sat: float = 25.0  # m³/h – half-saturation


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
    params: ContactorParams,
) -> ContactorResult:
    """Compute H₂S removal for current conditions.

    Args:
        gas_inlet: incoming gas conditions
        oxidant: current sump oxidant level (mg/L)
        alkalinity: current sump alkalinity (meq/L)
        recirc_flow: liquid recirculation rate (m³/h)
        params: contactor parameters
    """
    # ── Oxidant factor: Monod saturation ──────────────────────────
    f_oxidant = oxidant / (oxidant + params.oxidant_half_sat)

    # ── Alkalinity factor: Monod saturation ───────────────────────
    f_alkalinity = alkalinity / (alkalinity + params.alkalinity_half_sat)

    # ── Gas flow factor: power law ────────────────────────────────
    flow_ratio = gas_inlet.gas_flow / params.nominal_gas_flow
    f_gas = jnp.power(jnp.maximum(flow_ratio, 0.01), params.gas_flow_exponent)

    # ── Recirculation factor: Monod saturation ────────────────────
    f_recirc = recirc_flow / (recirc_flow + params.recirc_half_sat)

    # ── Weighted combination ──────────────────────────────────────
    # Chemistry factors (oxidant + alkalinity) and hydraulic factors (gas, recirc)
    chem_factor = (params.oxidant_weight * f_oxidant + params.alkalinity_weight * f_alkalinity) / (params.oxidant_weight + params.alkalinity_weight)

    hydraulic_factor = f_gas * ((1.0 - params.recirc_weight) + params.recirc_weight * f_recirc)

    efficiency = params.base_efficiency * chem_factor * hydraulic_factor
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
