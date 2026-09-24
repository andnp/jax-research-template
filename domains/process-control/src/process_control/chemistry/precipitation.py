"""FeCl₃ phosphate precipitation chemistry.

Models the removal of dissolved phosphate (PO₄-P) by dosing ferric
chloride (FeCl₃).  The reaction is simplified to a single-step
stoichiometric removal with Monod-type saturation:

    Fe³⁺ + PO₄³⁻ → FePO₄ ↓

At low Fe:P molar ratios the removal is nearly stoichiometric.  At
high ratios, diminishing returns set in as competing reactions consume
excess iron (hydroxide formation, organic complexation).

The model tracks residual dissolved PO₄-P after precipitation:

    removed_P = dose_Fe / stoich_fe_per_p * (P / (P + K_half))
    P_out = max(P_in - removed_P, P_min)

``stoich_fe_per_p`` is a *practical* (not theoretical 1:1) molar Fe:P
ratio: it already folds in the fraction of dosed iron lost to side
reactions such as hydroxide formation and organic complexation, so it
should not be multiplied by a further "efficiency" factor on top (that
double-counts the same loss). Reported practical ratios for effluent
P below ~1 mg/L cluster around 1.5-3 mol Fe/mol P; see Szabó et al.
(2008), "Significance of Design and Operational Variables in Chemical
Phosphorus Removal", Water Environment Research 80(5):407-416,
https://doi.org/10.2175/106143008X268498, and WEF Manual of Practice
No. 8 for the general dosing guidance this reduced-order model follows.

FeCl₃ hydrolysis also consumes alkalinity (Fe³⁺ + 3 HCO₃⁻ → Fe(OH)₃ +
3 CO₂ + 3 Cl⁻, i.e. ``alk_per_fe`` ≈ 3 eq/mol Fe); this module does not
track pH/alkalinity itself, so callers that carry an alkalinity state
(e.g. the ASM2d-based combined N/P benchmark) should apply that
consumption themselves using ``PrecipitationParams.alk_per_fe``.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PrecipitationParams:
    stoich_fe_per_p: float = 1.8  # practical mol Fe / mol P (Szabó et al. 2008; already nets out side reactions)
    k_half: float = 0.5  # mg-P/L half-saturation for Monod uptake
    p_min: float = 0.01  # mg-P/L floor (residual non-reactive P)
    mw_fe: float = 55.85  # g/mol (molar mass of Fe)
    mw_p: float = 30.97  # g/mol (molar mass of P)
    alk_per_fe: float = 3.0  # eq alkalinity consumed per mol Fe dosed (FeCl₃ hydrolysis)


def precipitate(
    influent_p: jax.Array,
    fe_dose: jax.Array,
    params: PrecipitationParams,
):
    """Compute residual PO₄-P after FeCl₃ dosing.

    Args:
        influent_p: inlet PO₄-P concentration (mg-P/L)
        fe_dose: FeCl₃ dose as mg-Fe/L
        params: precipitation parameters

    Returns:
        (effluent_p, fe_consumed)
        - effluent_p: residual PO₄-P (mg-P/L)
        - fe_consumed: iron consumed in precipitation (mg-Fe/L)
    """
    # Convert Fe dose to moles, then to equivalent P removal capacity
    fe_moles = fe_dose / params.mw_fe
    p_removal_capacity = fe_moles * params.mw_p / params.stoich_fe_per_p

    # Monod saturation: removal limited when P is low
    monod = influent_p / (influent_p + params.k_half)
    actual_removal = jnp.minimum(p_removal_capacity * monod, influent_p - params.p_min)
    actual_removal = jnp.maximum(actual_removal, 0.0)

    effluent_p = influent_p - actual_removal
    fe_consumed = actual_removal / params.mw_p * params.mw_fe * params.stoich_fe_per_p

    return effluent_p, fe_consumed
