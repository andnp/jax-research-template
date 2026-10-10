"""BSM1 performance indices: effluent quality, limit compliance and operating cost.

Pure functions over trajectories (arrays indexed by time), following the
performance-assessment definitions in Alex et al. (2008) §6:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf

All integrals in the reference are rectangular (constant time step), so a
time integral divided by the observation period T reduces to a plain mean;
none of these functions need the step size as an argument. Sludge production
(SP) and external-carbon consumption (EC) require a solids/dosing inventory
that most benchmarks here don't track (perfect-clarifier assumption, no
carbon dosing action), so `operational_cost_index` takes them as optional
inputs defaulting to zero rather than approximating them.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.units.asm1 import ASM1Params, ASM1State, compute_bod5, compute_cod, compute_tkn, compute_tss


@dataclass(frozen=True)
class EQIWeights:
    """Pollution-unit weights (g pollution unit / g), Alex et al. (2008) Table 10."""

    b_ss: float = 2.0
    b_cod: float = 1.0
    b_nkj: float = 30.0
    b_no: float = 10.0
    b_bod5: float = 2.0


def effluent_quality_index(
    effluent: ASM1State,
    params: ASM1Params,
    q_e: jax.Array,
    weights: EQIWeights | None = None,
) -> jax.Array:
    """Time-averaged Effluent Quality Index (kg pollution units/d).

    `effluent` fields and `q_e` (m³/d) share a leading time axis.
    """
    if weights is None:
        weights = EQIWeights()
    ss = compute_tss(effluent)
    cod = compute_cod(effluent)
    bod5 = compute_bod5(effluent, params)
    tkn = compute_tkn(effluent, params)
    pollution_load = (
        weights.b_ss * ss + weights.b_cod * cod + weights.b_nkj * tkn + weights.b_no * effluent.s_no + weights.b_bod5 * bod5
    ) * q_e
    return jnp.mean(pollution_load) / 1000.0


@dataclass(frozen=True)
class EffluentLimits:
    """Effluent quality limits (g/m³), Alex et al. (2008) Table 9."""

    n_tot: float = 18.0
    cod: float = 100.0
    s_nh: float = 4.0
    tss: float = 30.0
    bod5: float = 10.0

    def items(self) -> tuple[tuple[str, float], ...]:
        return (("n_tot", self.n_tot), ("cod", self.cod), ("s_nh", self.s_nh), ("tss", self.tss), ("bod5", self.bod5))


def time_above_limit(concentration: jax.Array, limit: float) -> jax.Array:
    """Fraction of the evaluation time an effluent concentration, sampled at a constant step, exceeds its limit."""
    return jnp.mean(concentration > limit)


def limit_violations(concentration: jax.Array, limit: float) -> jax.Array:
    """Number of violations: crossings of the limit from below to above, along the leading time axis."""
    above = concentration > limit
    return jnp.sum(above[1:] & ~above[:-1])


def percentile_95(concentration: jax.Array) -> jax.Array:
    """The concentration exceeded 5 % of the evaluation time, as BSM1 reports for S_NH, N_tot and TSS."""
    return jnp.percentile(concentration, 95.0)


def aeration_energy(kla_per_day: jax.Array, volumes: jax.Array, s_o_sat: float = 8.0) -> jax.Array:
    """Time-averaged aeration energy AE (kWh/d).

    `kla_per_day` has shape (T, n_reactors) in d⁻¹ (not h⁻¹); `volumes` (m³)
    has shape (n_reactors,).
    """
    kla_volume = jnp.sum(kla_per_day * volumes, axis=-1)
    return s_o_sat / (1.8 * 1000.0) * jnp.mean(kla_volume)


def pumping_energy(q_a: jax.Array, q_r: jax.Array, q_w: jax.Array) -> jax.Array:
    """Time-averaged pumping energy PE (kWh/d) for internal recycle, return sludge, and waste flows (m³/d)."""
    return jnp.mean(0.004 * q_a + 0.008 * q_r + 0.05 * q_w)


def mixing_energy(volumes: jax.Array, kla_per_day: jax.Array, unmixed_threshold: float = 20.0) -> jax.Array:
    """Time-averaged mixing energy ME (kWh/d).

    Any compartment with KLa below `unmixed_threshold` (d⁻¹) needs mechanical
    mixing to keep solids in suspension; aerated compartments normally clear
    the threshold and contribute zero. `volumes`/`kla_per_day` cover all
    reactors, shapes matching `aeration_energy`.
    """
    needs_mixing = kla_per_day < unmixed_threshold
    mixing_load = jnp.sum(jnp.where(needs_mixing, 0.005 * volumes, 0.0), axis=-1)
    return 24.0 * jnp.mean(mixing_load)


def sludge_production(solids_start: jax.Array, solids_end: jax.Array, waste_solids: jax.Array, period: float) -> jax.Array:
    """Sludge production SP (kg SS/d) over `period` days.

    The change in solids held in the reactors and settler (kg SS) from start to end of the period, plus
    the time-averaged wasted solids flow `waste_solids` (kg SS/d).
    """
    return (solids_end - solids_start) / period + jnp.mean(waste_solids)


def operational_cost_index(
    ae: jax.Array,
    pe: jax.Array,
    sp: jax.Array | float = 0.0,
    ec: jax.Array | float = 0.0,
    me: jax.Array | float = 0.0,
) -> jax.Array:
    """Operational Cost Index OCI = AE + PE + 5·SP + 3·EC + ME (Alex et al., 2008, §6)."""
    return ae + pe + 5.0 * sp + 3.0 * ec + me
