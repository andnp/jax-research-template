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
    # 0-indexed layer where feed enters. BSM1 places the feed at the "6th
    # layer counting from bottom to top" (1-indexed m=6), i.e. index 5 here:
    # 5 layers below (thickening zone) + feed + 4 layers above (clarification
    # zone). Alex et al. (2008), "Benchmark Simulation Model no. 1 (BSM1)",
    # Sec. 2.3.3, p.8. https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
    feed_layer: int = 5
    v_0_max: float = 10.417  # practical max settling velocity (m/h; 250 m/d)
    v_0: float = 19.75  # Vesilind max settling velocity (m/h; 474 m/d)
    r_h: float = 0.000576  # hindered settling parameter (m³/g)
    r_p: float = 0.00286  # flocculant settling parameter (m³/g)
    f_ns: float = 0.00228  # non-settleable fraction of feed TSS
    # Threshold above which a layer is treated as "already thickened" when
    # deciding whether the clarification zone flux limiter applies (BSM1
    # Table 4 / Sec. 2.3.3, p.10: "the threshold concentration X_t is equal
    # to 3,000 g.m⁻³"). Same source as feed_layer above.
    x_t: float = 3000.0


@jax_dataclass
class TakacsSettlerState:
    """State of the 10-layer Takács settler.

    layer_tss: TSS concentration in each layer (g/m³), shape (n_layers,).
    Layer 0 is the bottom (thickening zone), layer n-1 is the top (effluent).
    """

    layer_tss: jax.Array


def reset(feed_tss: float | jax.Array, params: TakacsSettlerParams, rng_key: jax.Array) -> TakacsSettlerState:
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
    """Advance the settler by one timestep of forward Euler.

    Explicit Euler is stable here only while dt stays below roughly
    h_layer / v_0_max (the CFL limit for the fastest advective/settling
    signal crossing one layer); at the BSM1 defaults (h_layer=0.4 m,
    v_0_max=10.417 m/h) that bound is ~0.038 h, so dt=0.02 h carries
    limited margin. This is a numerical-stability argument, not a cited
    BSM1 requirement — the BSM1 report leaves solver/step-size choice to
    the implementer.
    """
    n = params.n_layers
    j_f = params.feed_layer
    h_layer = params.depth / n
    vol_layer = params.area * h_layer

    X = state.layer_tss
    x_min = params.f_ns * feed_tss
    q_effluent = q_feed - q_underflow

    # Settling velocity and unlimited flux for each layer
    v_s = _settling_velocity(X, x_min, params)
    J_s = v_s * X  # settling flux (g / m² / h), downward

    # --- Settling contribution, with the Takács flux limiter ---
    # At each of the n-1 inter-layer boundaries (boundary j sits between
    # layer j and layer j-1 below it), the downward flux cannot exceed what
    # the receiving layer below can itself pass on: min(J_s[j], J_s[j-1]).
    # Below and at the feed layer this cap always applies; above the feed
    # layer (clarification zone) it only applies once the layer below the
    # boundary has thickened past x_t — otherwise the local unlimited flux
    # is used, since a genuinely clear layer isn't limited by a denser one
    # further down. Alex et al. (2008) BSM1 report, Sec. 2.3.3, pp. 9-10.
    boundary_j = jnp.arange(1, n, dtype=jnp.float32)
    J_s_upper = J_s[1:]
    J_s_lower = J_s[:-1]
    X_lower = X[:-1]
    limited = jnp.minimum(J_s_upper, J_s_lower)
    is_thickening = boundary_j <= j_f
    flux_boundary = jnp.where(is_thickening, limited, jnp.where(X_lower > params.x_t, limited, J_s_upper))

    J_s_from_above = jnp.concatenate([flux_boundary, jnp.array([0.0])])
    J_s_leaving = jnp.concatenate([jnp.array([0.0]), flux_boundary])
    settling = (J_s_from_above - J_s_leaving) / h_layer

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
