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
    # Peak capture at the optimum polymer dose. Belt-press polymer-aided capture is
    # ~90-98% (Metcalf & Eddy sludge dewatering tables); 0.97 is a calibrated
    # benchmark choice, not a site value.
    max_capture: float = 0.97
    # Dose practice is per unit dry solids, not per unit sludge volume: belt
    # presses run ~2-10 kg polymer/tonne DS, centrifuges ~8-15 kg/tonne DS
    # (WEF MOP 8; cited link no longer resolves — unverified against the
    # primary manual, though broadly consistent with other industry dosing
    # references for belt presses and centrifuges).
    k_polymer_capture: float = 3.0  # half-sat dose ratio for capture (kg/tonne DS)
    # Overdosed polymer restabilises floc / blinds the media, reducing
    # capture past an optimum — a Haldane-type (substrate-inhibition) response.
    # ki_polymer_capture sets that optimum (~sqrt(k*ki) kg/tonne DS); reduced-order,
    # numeric value unverified against sludge-specific overdose literature.
    ki_polymer_capture: float = 30.0  # overdose inhibition scale (kg/tonne DS)
    base_dryness: float = 0.18  # cake dry solids fraction at zero polymer
    max_dryness: float = 0.30  # cake dry solids fraction at saturating polymer
    k_polymer_dryness: float = 4.0  # half-sat dose ratio for dryness (kg/tonne DS)
    speed_penalty: float = 0.15  # capture loss per unit normalised speed above 0.5
    # Faster belt/bowl travel shortens dewatering (pressing/centrifugation)
    # residence time, so cake retains more moisture — qualitative direction
    # only, magnitude unverified against a specific press/centrifuge source.
    dryness_speed_penalty: float = 0.05  # dryness fraction lost per unit speed above 0.5


@jax_dataclass
class DewateringState:
    cake_produced: jax.Array  # cumulative cake mass (kg dry solids)
    filtrate_volume: jax.Array  # cumulative filtrate volume (m³)
    # Feed above belt/centrifuge capacity (q_feed - q_actual) is neither
    # processed nor discarded here — it accumulates as backlog so the unit
    # doesn't silently destroy mass; a caller decides what to do with it
    # (hold it, alarm on it, etc.).
    backlog_volume: jax.Array  # cumulative unprocessed feed volume (m³)
    backlog_solids: jax.Array  # cumulative unprocessed solids mass (kg)


def reset(params: DewateringParams, rng_key: jax.Array):
    return DewateringState(
        cake_produced=jnp.array(0.0),
        filtrate_volume=jnp.array(0.0),
        backlog_volume=jnp.array(0.0),
        backlog_solids=jnp.array(0.0),
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
        (new_state, cake_dryness, filtrate_tss, q_filtrate, bypass_flow)
    """
    # Actual throughput limited by belt speed; anything above that bypasses
    # the unit untreated rather than vanishing (see DewateringState.backlog_*).
    q_actual = jnp.minimum(q_feed, belt_speed * params.max_throughput)
    bypass_flow = q_feed - q_actual

    # Dose ratio (kg polymer / tonne dry solids): polymer_dose and feed_tss
    # are both mass concentrations (mg/L == g/m³), so their ratio times 1000
    # converts directly to the practice-standard kg/tonne-DS unit — it does
    # not depend on q_actual, since both dose and solids scale with flow.
    dose_ratio = 1000.0 * polymer_dose / (feed_tss + 1e-10)

    # Capture: Haldane-type saturation-then-inhibition (overdose restabilises
    # floc / blinds the media past an optimum dose ratio).
    # The Haldane term peaks below 1 at sqrt(k*ki); normalise so max_capture is the true peak.
    haldane_peak = 1.0 / (1.0 + 2.0 * jnp.sqrt(params.k_polymer_capture / params.ki_polymer_capture))
    capture_response = dose_ratio / (
        params.k_polymer_capture + dose_ratio + dose_ratio**2 / params.ki_polymer_capture
    )
    capture = params.base_capture + (params.max_capture - params.base_capture) * capture_response / haldane_peak
    # Dryness keeps a plain Monod saturation — no reported overdose penalty on dryness.
    dryness = params.base_dryness + (params.max_dryness - params.base_dryness) * (dose_ratio / (params.k_polymer_dryness + dose_ratio))

    # Speed penalty: running faster reduces both capture and cake dryness
    speed_excess = jnp.maximum(belt_speed - 0.5, 0.0)
    capture = jnp.clip(capture - params.speed_penalty * speed_excess, 0.0, 1.0)
    dryness = jnp.clip(dryness - params.dryness_speed_penalty * speed_excess, 0.0, params.max_dryness)

    # Mass balance
    solids_in = feed_tss * q_actual * dt / 1e3  # kg (feed_tss in g/m³)
    solids_captured = solids_in * capture
    solids_lost = solids_in - solids_captured

    # Cake volumetric flow: dry-solids mass flow divided by (dryness fraction
    # * cake bulk density), not multiplied — a cake at lower dryness carries
    # more water per unit dry solids, so it takes a *larger* volume to remove
    # the same captured mass. Bulk density approximated as water's (1e6 g/m³);
    # filtrate is whatever volume isn't carried out in the cake.
    q_cake = q_actual * feed_tss * capture / (dryness * 1e6 + 1e-10)
    q_filtrate = jnp.maximum(q_actual - q_cake, 0.0)
    filtrate_tss = jnp.where(
        q_filtrate > 0.0,
        solids_lost * 1e3 / (q_filtrate * dt + 1e-10),  # g/m³
        0.0,
    )

    bypass_solids = feed_tss * bypass_flow * dt / 1e3  # kg, unprocessed

    new_state = DewateringState(
        cake_produced=state.cake_produced + solids_captured,
        filtrate_volume=state.filtrate_volume + q_filtrate * dt,
        backlog_volume=state.backlog_volume + bypass_flow * dt,
        backlog_solids=state.backlog_solids + bypass_solids,
    )

    return new_state, dryness, filtrate_tss, q_filtrate, bypass_flow
