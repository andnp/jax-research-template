from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Henson, M. A. & Seborg, D. E. (1994), "Adaptive Nonlinear Control of a pH
# Neutralization Process," IEEE Trans. Control Syst. Technol. 2(3), 169-182:
# https://sites.chemengr.ucsb.edu/~ceweb/faculty/seborg/cv/pub_pdfs/134.pdf
#
# The reaction-invariant charge balance (their eq. 9) ties pH to two
# conserved quantities: Wa (net H+ excess) and Wb (total dissolved carbonate).
# Both mix linearly under dilution, which is why our "base_excess" (= -Wa, so
# that adding base makes it more positive) can be carried as a plain CSTR
# concentration state. In pH space (their eq. 23, y = pH, x1 = Wa, x2 = Wb):
#   F(y) = -base_excess + 10^(y-pKw) - 10^(-y)
#          + Wb * (1 + 2*10^(y-pK2)) / (1 + 10^(pK1-y) + 10^(y-pK2))
# F is strictly increasing in y, so it has exactly one root; compute_ph
# solves F(y) = 0 with damped Newton iteration.
#
# Ka1 = 4.47e-7 (pKa1 = 6.35), Ka2 = 5.62e-11 (pKa2 = 10.25), Kw = 1.00e-14
# are the paper's Table I nominal values, consistent with standard 25°C
# carbonic-acid dissociation constants (e.g. Millero, "Dissociation Constants
# for Carbonic Acid Determined from Field Measurements," Deep-Sea Research I,
# 2002: https://www.gfdl.noaa.gov/bibliography/related_files/millero0201.pdf).
PKA1 = 6.35
PKA2 = 10.25
PKW = 14.0

_NEWTON_ITERS = 15
_MAX_STEP = 2.0  # pH units per Newton step
_PH_BOUNDS = (-5.0, 21.0)  # keeps 10**(...) terms from overflowing float32


@dataclass(frozen=True)
class PhModelParams:
    # Wb: total dissolved carbonate (mol/L), the wastewater's fixed background
    # buffering capacity. 0.005 mol/L (~5 meq/L) corresponds to ~250 mg/L
    # alkalinity as CaCO3; the cited source discusses municipal wastewater
    # alkalinity requirements but does not itself state a 250-334 mg/L
    # typical range, so that specific range is unverified:
    # https://mantech-inc.com/wp-content/uploads/2018/07/Alkalinity-Profiling-in-Wastewater-Operations.pdf
    buffer_capacity: float = 0.005


def _charge_balance_and_slope(
    ph: jax.Array,
    base_excess: jax.Array,
    buffer_capacity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate F(pH) and dF/dpH from the Henson & Seborg charge balance."""
    ln10 = jnp.log(10.0)
    term_oh = 10.0 ** (ph - PKW)
    term_h = 10.0 ** (-ph)
    hco3_weight = 10.0 ** (PKA1 - ph)
    co3_weight = 10.0 ** (ph - PKA2)

    numerator = 1.0 + 2.0 * co3_weight
    denominator = 1.0 + hco3_weight + co3_weight
    buffer_term = buffer_capacity * numerator / denominator

    # Henson & Seborg's Wa is signed as net H+ excess (acidic > 0); our
    # base_excess is the opposite sign (net base excess, basic > 0), so the
    # invariant enters the balance as -base_excess rather than +Wa.
    residual = term_oh - term_h + buffer_term - base_excess

    d_hco3_weight = -ln10 * hco3_weight
    d_co3_weight = ln10 * co3_weight
    d_numerator = 2.0 * d_co3_weight
    d_denominator = d_hco3_weight + d_co3_weight
    d_buffer_term = buffer_capacity * (d_numerator * denominator - numerator * d_denominator) / denominator**2

    slope = ln10 * term_oh + ln10 * term_h + d_buffer_term
    return residual, slope


def compute_ph(base_excess: jax.Array, params: PhModelParams) -> jax.Array:
    """Solve the reaction-invariant charge balance for pH given net acid/base charge.

    Starts from neutral pH and refines with a fixed number of step-clamped
    Newton iterations, so the routine is jit/vmap-safe and needs no
    data-dependent control flow. Clamping the step size matters near the
    buffered plateau, where the slope is small enough that an undamped
    Newton step overshoots into the region where the 10**(...) terms
    overflow float32.
    """
    base_excess = jnp.asarray(base_excess)
    buffer_capacity = jnp.asarray(params.buffer_capacity)
    ph = jnp.full_like(base_excess, 7.0)

    for _ in range(_NEWTON_ITERS):
        residual, slope = _charge_balance_and_slope(ph, base_excess, buffer_capacity)
        step = jnp.clip(residual / slope, -_MAX_STEP, _MAX_STEP)
        ph = jnp.clip(ph - step, *_PH_BOUNDS)

    return ph
