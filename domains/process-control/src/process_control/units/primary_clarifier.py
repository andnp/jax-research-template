"""Primary clarifier unit model.

A simplified gravity settler for primary treatment. Removes TSS as a
function of hydraulic retention time (HRT), using the empirical
Crites & Tchobanoglous / Metcalf & Eddy correlation:

    R(t) = t / (a + b*t)

with a=0.0075 h, b=0.014 h giving ~59% TSS removal at a 2.5 h HRT, in
the observed 50-70% range for primary clarifiers (same a/b values and
worked example reproduced from Metcalf & Eddy's design tables in
Ghangrekar, "Primary Sedimentation Tank" (IIT Kharagpur course notes),
http://ghangrekar.com/wp-content/uploads/2016/02/11-Sedimentation-Final.pdf).

Sludge accumulates at the bottom and must be wasted to prevent
inventory build-up. The underflow TSS concentration depends on
current sludge inventory.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class PrimaryClarifierParams:
    area: float = 500.0  # settling area (m²)
    depth: float = 3.0  # settler depth (m)
    a_settle: float = 0.0075  # Crites & Tchobanoglous TSS removal constant (h)
    b_settle: float = 0.014  # Crites & Tchobanoglous TSS removal constant (h)
    max_sludge_mass: float = 5e6  # g — capacity limit (triggers overflow)
    # Primary sludge is typically 2-6% total solids (Metcalf & Eddy,
    # Wastewater Engineering: Treatment and Resource Recovery); 5000 g/m³
    # (0.5%) was below that whole range.
    min_underflow_tss: float = 20000.0  # g/m³ minimum underflow concentration (2% TS)
    max_underflow_tss: float = 60000.0  # g/m³ maximum underflow concentration (6% TS)


@jax_dataclass
class PrimaryClarifierState:
    sludge_mass: jax.Array  # accumulated sludge (g)


def reset(params: PrimaryClarifierParams, rng_key: jax.Array):
    initial_mass = params.max_sludge_mass * 0.3
    return PrimaryClarifierState(sludge_mass=jnp.array(initial_mass))


def step(
    state: PrimaryClarifierState,
    feed_tss: jax.Array,
    q_feed: jax.Array,
    q_waste: jax.Array,
    params: PrimaryClarifierParams,
    dt: jax.Array,
)-> tuple[PrimaryClarifierState, jax.Array, jax.Array, jax.Array]:
    """Advance primary clarifier by one timestep.

    Args:
        feed_tss: influent TSS (g/m³)
        q_feed: influent flow (m³/h)
        q_waste: primary sludge wastage rate (m³/h)
        dt: timestep (h)

    Returns:
        (new_state, effluent_tss, underflow_tss, overflow)
    """
    # R(t) = t/(a+b*t) yields percent removal directly, per the source table.
    hrt = params.area * params.depth / (q_feed + 1e-10)
    eta = hrt / (params.a_settle + params.b_settle * hrt) / 100.0

    tss_removed = eta * feed_tss
    effluent_tss = feed_tss - tss_removed

    # Sludge mass balance. mass_out is capped by what's actually in the
    # settler (mass_in included), so wasting more than is available can't
    # manufacture mass; overflow beyond capacity stays in the mass balance
    # by being routed into the effluent instead of deleted.
    mass_in = tss_removed * q_feed * dt  # g deposited
    underflow_tss = jnp.clip(
        state.sludge_mass / (params.area * params.depth) * 2.0,
        params.min_underflow_tss,
        params.max_underflow_tss,
    )
    available_mass = state.sludge_mass + mass_in
    mass_out = jnp.minimum(underflow_tss * q_waste * dt, available_mass)
    raw_mass = available_mass - mass_out
    overflow = jnp.maximum(raw_mass - params.max_sludge_mass, 0.0)
    new_mass = raw_mass - overflow
    effluent_tss = effluent_tss + overflow / (q_feed * dt + 1e-10)

    new_state = PrimaryClarifierState(sludge_mass=new_mass)
    return new_state, effluent_tss, underflow_tss, overflow
