"""Coagulation chemistry module.

Models coagulant (e.g. alum or ferric) injection for particle
destabilisation. The dose-response follows a re-stabilisation curve:

- Underdose: insufficient charge neutralisation → poor removal
- Optimal dose: maximum particle destabilisation
- Overdose: charge reversal → re-stabilisation, reduced removal

    eta(dose) = eta_max * (dose / dose_opt)^alpha * exp(1 - (dose / dose_opt)^alpha)

This gives a peaked curve with maximum at dose_opt, declining on both sides.
Influent turbidity and NOM affect the optimal dose.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CoagulationParams:
    dose_opt_base: float = 30.0  # optimal dose at reference turbidity (mg/L)
    eta_max: float = 0.90  # max TSS removal at optimal dose
    alpha: float = 2.0  # shape parameter for dose-response curve
    turbidity_sensitivity: float = 0.5  # dose_opt scales with turbidity^this
    ref_turbidity: float = 10.0  # reference turbidity (NTU) for dose_opt_base
    min_removal: float = 0.05  # minimum removal even at zero dose (settling)


def coagulate(
    feed_tss: jax.Array,
    feed_turbidity: jax.Array,
    coag_dose: jax.Array,
    params: CoagulationParams,
):
    """Compute post-coagulation TSS.

    Args:
        feed_tss: influent TSS (g/m³ = mg/L)
        feed_turbidity: influent turbidity (NTU)
        coag_dose: coagulant dose (mg/L)
        params: coagulation parameters

    Returns:
        (effluent_tss, removal_efficiency)
    """
    # Optimal dose adjusts with turbidity
    dose_opt = params.dose_opt_base * (feed_turbidity / params.ref_turbidity) ** params.turbidity_sensitivity

    # Peaked dose-response curve
    x = jnp.clip(coag_dose / (dose_opt + 1e-10), 1e-6, 10.0)
    eta = params.eta_max * (x**params.alpha) * jnp.exp(1.0 - x**params.alpha)
    eta = jnp.clip(eta, params.min_removal, params.eta_max)

    effluent_tss = feed_tss * (1.0 - eta)
    return effluent_tss, eta
