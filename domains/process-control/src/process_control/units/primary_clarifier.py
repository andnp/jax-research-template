"""Primary clarifier unit model.

A simplified gravity settler for primary treatment. Removes TSS as a
function of hydraulic retention time (HRT), using the empirical
Crites & Tchobanoglous / Metcalf & Eddy correlation:

    R(t) = t / (a + b*t)

with a=0.0075 h, b=0.014 h giving ~59% TSS removal at a 2.5 h HRT, in
the observed 50-70% range for primary clarifiers (values as widely
reproduced from Metcalf & Eddy's design tables, e.g.
https://d3pcsg2wjq9izr.cloudfront.net/files/28606/download/33719/SanJose.pdf).

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
    max_sludge_mass: float = 5e6  # kg — capacity limit (triggers overflow)
    min_underflow_tss: float = 5000.0  # g/m³ minimum underflow concentration


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

    # Sludge mass balance
    mass_in = tss_removed * q_feed * dt  # g deposited
    underflow_tss = jnp.maximum(
        state.sludge_mass / (params.area * params.depth) * 2.0,
        params.min_underflow_tss,
    )
    mass_out = underflow_tss * q_waste * dt  # g removed
    raw_mass = state.sludge_mass + mass_in - mass_out
    overflow = jnp.maximum(raw_mass - params.max_sludge_mass, 0.0)
    new_mass = jnp.clip(raw_mass, 0.0, params.max_sludge_mass)

    new_state = PrimaryClarifierState(sludge_mass=new_mass)
    return new_state, effluent_tss, underflow_tss, overflow
