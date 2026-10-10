import jax
import jax.numpy as jnp
from process_control.chemistry.species import HenrySolubility
from process_control.chemistry.thermo import Evaluation


def henry_hcp(record: HenrySolubility, temperature_k: jax.Array):
    """Henry solubility Hcp(T) in mol/(m3 Pa), with the temperature guard reported as in_domain.

    Hcp relates a gas partial pressure to the molecular aqueous concentration in dilute pure
    water. Only molecular species apply (H2S, NH3, CO2), never sulfide, ammonia or carbonate
    totals. For CO2 the record describes molecular CO2(aq), so using it for the CO2* pool
    assumes CO2* ~ CO2(aq), which is reasonable only at low temperature. Temperature is never
    clamped into the guard range.
    """
    t = jnp.asarray(temperature_k)
    value = record.hcp_ref_mol_per_m3_pa * jnp.exp(record.b_kelvin * (1.0 / t - 1.0 / record.t_ref_k))
    numerically_valid = (t > 0.0) & jnp.isfinite(t) & jnp.isfinite(value) & (value > 0.0)
    in_guard = (t >= record.guard_t_min_k) & (t <= record.guard_t_max_k)
    return Evaluation(value, numerically_valid, numerically_valid & in_guard)


def equilibrium_concentration(hcp: jax.Array, partial_pressure_pa: jax.Array):
    """Molecular aqueous concentration in mol/m3 in equilibrium with a partial pressure."""
    return hcp * partial_pressure_pa


def equilibrium_partial_pressure(hcp: jax.Array, molecular_concentration_mol_per_m3: jax.Array):
    """Partial pressure in Pa in equilibrium with a molecular aqueous concentration."""
    return molecular_concentration_mol_per_m3 / hcp
