"""FeCl₃ phosphate precipitation chemistry.

Models the removal of dissolved phosphate (PO₄-P) by dosing ferric
chloride (FeCl₃).  The reaction is simplified to a single-step
stoichiometric removal with Monod-type saturation:

    Fe³⁺ + PO₄³⁻ → FePO₄ ↓

At low Fe:P molar ratios the removal is nearly stoichiometric.  At
high ratios, diminishing returns set in as competing reactions consume
excess iron (hydroxide formation, organic complexation).

The model tracks residual dissolved PO₄-P after precipitation:

    removed_P = dose_Fe * (P / (P + K_half)) * efficiency
    P_out = max(P_in - removed_P, P_min)

where ``K_half`` sets the half-saturation for PO₄ availability and
``efficiency`` captures the fraction of dosed iron that actually
precipitates phosphate (vs. side reactions).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PrecipitationParams:
    stoich_fe_per_p: float = 1.8  # mol Fe / mol P (practical, > 1.0 theoretical)
    k_half: float = 0.5  # mg-P/L half-saturation for Monod uptake
    efficiency: float = 0.85  # fraction of Fe that precipitates P (vs. side reactions)
    p_min: float = 0.01  # mg-P/L floor (residual non-reactive P)
    mw_fe: float = 55.85  # g/mol (molar mass of Fe)
    mw_p: float = 30.97  # g/mol (molar mass of P)


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
    p_removal_capacity = fe_moles * params.mw_p / params.stoich_fe_per_p * params.efficiency

    # Monod saturation: removal limited when P is low
    monod = influent_p / (influent_p + params.k_half)
    actual_removal = jnp.minimum(p_removal_capacity * monod, influent_p - params.p_min)
    actual_removal = jnp.maximum(actual_removal, 0.0)

    effluent_p = influent_p - actual_removal
    fe_consumed = actual_removal / params.mw_p * params.mw_fe * params.stoich_fe_per_p / params.efficiency

    return effluent_p, fe_consumed
