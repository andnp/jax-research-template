"""Dewatering unit model (belt filter press / centrifuge).

Simplified model of mechanical sludge dewatering:
- Polymer dose improves cake dryness and solids capture
- Belt speed (or throughput) trades capacity against performance
- Filtrate returns to headworks with residual TSS

Dose-response uses a Monod-type saturation for both capture and dryness.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class DewateringParams:
    max_throughput: float = 50.0  # m³/h max sludge feed rate
    base_capture: float = 0.85  # capture rate at zero polymer
    max_capture: float = 0.98  # capture rate at saturating polymer
    k_polymer_capture: float = 3.0  # half-sat polymer for capture (mg/L)
    base_dryness: float = 0.18  # cake dry solids fraction at zero polymer
    max_dryness: float = 0.30  # cake dry solids fraction at saturating polymer
    k_polymer_dryness: float = 5.0  # half-sat polymer for dryness (mg/L)
    speed_penalty: float = 0.15  # capture loss per unit normalised speed above 0.5


@jax_dataclass
class DewateringState:
    cake_produced: jax.Array  # cumulative cake mass (kg dry solids)
    filtrate_volume: jax.Array  # cumulative filtrate volume (m³)


def reset(params: DewateringParams, rng_key: jax.Array):
    return DewateringState(
        cake_produced=jnp.array(0.0),
        filtrate_volume=jnp.array(0.0),
    )


def step(
    state: DewateringState,
    feed_tss: jax.Array,
    q_feed: jax.Array,
    polymer_dose: jax.Array,
    belt_speed: jax.Array,
    params: DewateringParams,
    dt: jax.Array,
):
    """Advance dewatering by one timestep.

    Args:
        feed_tss: sludge TSS concentration (g/m³)
        q_feed: sludge feed flow (m³/h)
        polymer_dose: polymer concentration (mg/L)
        belt_speed: normalised belt speed (0-1)
        dt: timestep (h)

    Returns:
        (new_state, cake_dryness, filtrate_tss, q_filtrate)
    """
    # Actual throughput limited by belt speed
    q_actual = jnp.minimum(q_feed, belt_speed * params.max_throughput)

    # Polymer-dependent capture and dryness (Monod saturation)
    capture = params.base_capture + (params.max_capture - params.base_capture) * (polymer_dose / (params.k_polymer_capture + polymer_dose))
    dryness = params.base_dryness + (params.max_dryness - params.base_dryness) * (polymer_dose / (params.k_polymer_dryness + polymer_dose))

    # Speed penalty: running faster reduces capture
    speed_penalty = params.speed_penalty * jnp.maximum(belt_speed - 0.5, 0.0)
    capture = jnp.clip(capture - speed_penalty, 0.0, 1.0)

    # Mass balance
    solids_in = feed_tss * q_actual * dt / 1e3  # kg (feed_tss in g/m³)
    solids_captured = solids_in * capture
    solids_lost = solids_in - solids_captured

    # Filtrate
    q_filtrate = q_actual * (1.0 - dryness * capture * feed_tss / 1e6)
    q_filtrate = jnp.maximum(q_filtrate, 0.0)
    filtrate_tss = jnp.where(
        q_filtrate > 0.0,
        solids_lost * 1e3 / (q_filtrate * dt + 1e-10),  # g/m³
        0.0,
    )

    new_state = DewateringState(
        cake_produced=state.cake_produced + solids_captured,
        filtrate_volume=state.filtrate_volume + q_filtrate * dt,
    )

    return new_state, dryness, filtrate_tss, q_filtrate
