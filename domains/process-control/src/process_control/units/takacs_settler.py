"""Takács 10-layer secondary clarifier model.

Implements the double-exponential settling velocity function from
Takács et al. (1991) with the 10-layer 1D flux model used in BSM1
(Copp et al., 2002).

References:
  Takács, I., Patry, G.G., Nolasco, D. (1991). A dynamic model of the
    clarification-thickening process. Water Research, 25(10), 1263-1271.
  Copp, J.B. (2002). The COST Simulation Benchmark.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class TakacsSettlerParams:
    """Parameters for the Takács 10-layer secondary clarifier.

    Default values follow the BSM1 specification at 15°C.
    Settling velocities converted from m/d to m/h.
    """

    area: float = 1500.0  # settler surface area (m²)
    depth: float = 4.0  # settler depth (m)
    n_layers: int = 10  # number of horizontal layers
    feed_layer: int = 5  # 0-indexed layer where feed enters (6th from bottom)
    v_0_max: float = 10.417  # practical max settling velocity (m/h; 250 m/d)
    v_0: float = 19.75  # Vesilind max settling velocity (m/h; 474 m/d)
    r_h: float = 0.000576  # hindered settling parameter (m³/g)
    r_p: float = 0.00286  # flocculant settling parameter (m³/g)
    f_ns: float = 0.00228  # non-settleable fraction of feed TSS


@jax_dataclass
class TakacsSettlerState:
    """State of the 10-layer Takács settler.

    layer_tss: TSS concentration in each layer (g/m³), shape (n_layers,).
    Layer 0 is the bottom (thickening zone), layer n-1 is the top (effluent).
    """

    layer_tss: jax.Array


def reset(feed_tss: float, params: TakacsSettlerParams, rng_key: jax.Array) -> TakacsSettlerState:
    """Initialise settler with an approximate steady-state TSS profile.

    Below the feed layer: linearly increasing from feed_tss to 2× feed_tss.
    At the feed layer: feed_tss.
    Above the feed layer: exponentially decreasing toward zero.
    """
    n = params.n_layers
    j_f = params.feed_layer
    indices = jnp.arange(n, dtype=jnp.float32)

    # Thickening zone (below feed): linear increase toward bottom
    thickening = feed_tss * (1.0 + (j_f - indices) / jnp.maximum(j_f, 1.0))

    # Clarification zone (above feed): exponential decay toward top
    layers_above = indices - j_f
    clarification = feed_tss * jnp.exp(-2.0 * layers_above / jnp.maximum(n - j_f - 1, 1.0))

    # Combine: use thickening for j <= j_f, clarification for j > j_f
    profile = jnp.where(indices <= j_f, thickening, clarification)

    return TakacsSettlerState(layer_tss=profile)


def _settling_velocity(tss: jax.Array, x_min: jax.Array, params: TakacsSettlerParams) -> jax.Array:
    """Compute Takács settling velocity for each layer.

    v_s = min(v_0_max, max(0, v_0 * (exp(-r_h * dX) - exp(-r_p * dX))))
    where dX = max(0, X - X_min).
    """
    dx = jnp.maximum(tss - x_min, 0.0)
    v_s = params.v_0 * (jnp.exp(-params.r_h * dx) - jnp.exp(-params.r_p * dx))
    return jnp.minimum(params.v_0_max, jnp.maximum(0.0, v_s))


def step(
    state: TakacsSettlerState,
    feed_tss: jax.Array,
    q_feed: jax.Array,
    q_underflow: jax.Array,
    params: TakacsSettlerParams,
    dt: jax.Array,
) -> TakacsSettlerState:
    """Advance the settler by one timestep.

    Args:
        state: current settler state
        feed_tss: TSS concentration of feed (g/m³)
        q_feed: total feed flow rate (m³/h)
        q_underflow: total underflow rate (m³/h), Q_u = Q_rs + Q_w
        params: settler parameters
        dt: timestep (h)

    Returns:
        Updated settler state.
    """
    n = params.n_layers
    j_f = params.feed_layer
    h_layer = params.depth / n
    vol_layer = params.area * h_layer

    X = state.layer_tss
    x_min = params.f_ns * feed_tss
    q_effluent = q_feed - q_underflow

    # Settling velocity and flux for each layer
    v_s = _settling_velocity(X, x_min, params)
    J_s = v_s * X  # settling flux (g / m² / h), downward

    # --- Settling contribution ---
    # Flux arriving from layer above: J_s[j+1] for j < n-1, 0 for top layer
    J_s_from_above = jnp.concatenate([J_s[1:], jnp.array([0.0])])
    # Flux leaving downward: J_s[j] for j > 0, 0 for bottom (no layer below)
    J_s_leaving = X.at[0].set(0.0)  # just used to zero index 0
    J_s_leaving = J_s.at[0].set(0.0)
    settling = (J_s_from_above - J_s_leaving) * params.area / vol_layer

    # --- Convection contribution ---
    indices = jnp.arange(n, dtype=jnp.float32)

    # Above feed: upward flow Q_e carries X from layer below
    X_below = jnp.concatenate([jnp.array([0.0]), X[:-1]])
    conv_above = q_effluent * (X_below - X) / vol_layer

    # Below feed: downward flow Q_u carries X from layer above
    X_above = jnp.concatenate([X[1:], jnp.array([0.0])])
    conv_below = q_underflow * (X_above - X) / vol_layer

    # Feed layer: feed flow enters
    conv_feed = q_feed * (feed_tss - X) / vol_layer

    # Select convection term by zone
    is_above = indices > j_f
    is_feed = indices == j_f
    conv = jnp.where(is_above, conv_above, jnp.where(is_feed, conv_feed, conv_below))

    # --- Forward Euler ---
    dXdt = conv + settling
    new_X = jnp.maximum(0.0, X + dXdt * dt)

    return TakacsSettlerState(layer_tss=new_X)


def get_effluent_tss(state: TakacsSettlerState) -> jax.Array:
    """TSS in the top layer (effluent quality)."""
    return state.layer_tss[-1]


def get_underflow_tss(state: TakacsSettlerState) -> jax.Array:
    """TSS in the bottom layer (return sludge / waste sludge concentration)."""
    return state.layer_tss[0]


def compute_blanket_height(
    state: TakacsSettlerState,
    params: TakacsSettlerParams,
    threshold: float = 1500.0,
) -> jax.Array:
    """Estimate sludge blanket height as the depth of the highest layer above threshold.

    Returns height in metres from bottom. A value near params.depth indicates
    the blanket has risen to the top (imminent solids washout).
    """
    h_layer = params.depth / params.n_layers
    above_threshold = (state.layer_tss > threshold).astype(jnp.float32)
    # Number of layers (from bottom) that are above threshold
    blanket_layers = jnp.sum(above_threshold)
    return blanket_layers * h_layer
