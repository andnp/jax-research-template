"""One-dimensional layered settler with the Takács double-exponential settling velocity.

Takács, Patry & Nolasco (1991), "A dynamic model of the clarification-thickening process",
Water Research 25(10); as configured for BSM1 in Alex et al. (2008), Sec. 2.3.3:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control.components import ComponentSchema
from process_control.streams import Stream
from process_control.units.contract import Outlet


@dataclass(frozen=True)
class SettlerParams:
    """Settler geometry and Takács settling parameters; lengths in m, time in h, solids in g TSS/m³.

    `feed_layer` counts from 0 at the bottom. Settling velocity is
    clip(v0 (e^(−r_h ΔX) − e^(−r_p ΔX)), 0, v0_max) with ΔX = X − f_ns X_feed. Above the feed layer the
    flux leaving a layer is limited by the layer below only once that layer is thicker than x_t.
    """

    n_layers: int
    feed_layer: int
    area: float
    height: float
    v0_max: float
    v0: float
    r_h: float
    r_p: float
    f_ns: float
    x_t: float

    def __post_init__(self) -> None:
        if not 0 <= self.feed_layer < self.n_layers:
            raise ValueError(f"feed layer {self.feed_layer} is outside the {self.n_layers} layers")


BSM1_SETTLER = SettlerParams(
    n_layers=10,
    feed_layer=5,
    area=1500.0,
    height=4.0,
    v0_max=250.0 / 24.0,
    v0=474.0 / 24.0,
    r_h=0.000576,
    r_p=0.00286,
    f_ns=0.00228,
    x_t=3000.0,
)


@dataclass(frozen=True)
class LayeredSettler:
    """Non-reactive layered settler carrying every component per layer, bottom layer first.

    State has shape (n_layers, n_components). Solubles move with the bulk flow only; particulates also
    settle with their layer's solids flux, each in proportion to its share of the layer's TSS. BSM1
    instead gives both outlets the feed's particulate fractions, which propagates fraction changes
    instantly; carrying the components delays them by the settler's retention and keeps the outlets
    independent of the feed, while agreeing with BSM1 at steady state.

    The underflow flow is the `q_underflow` input (m³/h) and the effluent takes the rest of the feed,
    clamped at zero.
    """

    schema: ComponentSchema

    @property
    def inlet_ports(self) -> Mapping[str, ComponentSchema]:
        return {"feed": self.schema}

    @property
    def outlet_ports(self) -> Mapping[str, Outlet]:
        return {"effluent": Outlet(self.schema, flow_from=frozenset({"feed"})), "underflow": Outlet(self.schema)}

    @property
    def input_ports(self) -> tuple[str, ...]:
        return ("q_underflow",)

    def derivatives(self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: SettlerParams) -> jax.Array:
        feed = inlets["feed"]
        flows = self.outlet_flows(state, {"feed": feed.flow}, inputs, params)
        layer_height = params.height / params.n_layers
        return (self._advection(state, feed, flows, params) + self._settling(state, feed, params)) / layer_height

    def outlet_flows(
        self, state: jax.Array, inlet_flows: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array], params: SettlerParams
    ) -> Mapping[str, jax.Array]:
        underflow = jnp.maximum(inputs["q_underflow"], 0.0)
        return {"effluent": jnp.maximum(inlet_flows["feed"] - underflow, 0.0), "underflow": underflow}

    def outlet_compositions(
        self, state: jax.Array, inlets: Mapping[str, Stream], inputs: Mapping[str, jax.Array], params: SettlerParams
    ) -> Mapping[str, jax.Array]:
        return {"effluent": state[-1], "underflow": state[0]}

    def project(self, state: jax.Array, params: SettlerParams) -> jax.Array:
        return jnp.maximum(state, 0.0)

    def _advection(self, state: jax.Array, feed: Stream, flows: Mapping[str, jax.Array], params: SettlerParams) -> jax.Array:
        """Bulk transport flux per layer: up above the feed layer, down below it."""
        up = flows["effluent"] / params.area
        down = flows["underflow"] / params.area
        empty = jnp.zeros((1, state.shape[1]))
        from_below = jnp.concatenate([empty, state[:-1]])
        from_above = jnp.concatenate([state[1:], empty])

        layer = jnp.arange(params.n_layers)[:, None]
        rising = up * (from_below - state)
        sinking = down * (from_above - state)
        fed = feed.flow / params.area * feed.components - (up + down) * state
        return jnp.where(layer > params.feed_layer, rising, jnp.where(layer < params.feed_layer, sinking, fed))

    def _settling(self, state: jax.Array, feed: Stream, params: SettlerParams) -> jax.Array:
        """Gravity flux per layer, apportioned to particulates by their share of each layer's solids."""
        tss = self.schema.tss(state)
        excess = jnp.maximum(tss - params.f_ns * self.schema.tss(feed.components), 0.0)
        velocity = jnp.clip(params.v0 * (jnp.exp(-params.r_h * excess) - jnp.exp(-params.r_p * excess)), 0.0, params.v0_max)
        flux = velocity * tss

        # Boundary b lies between layer b + 1 above and layer b below.
        boundary = jnp.arange(1, params.n_layers)
        limited = jnp.minimum(flux[1:], flux[:-1])
        thickened = (boundary <= params.feed_layer) | (tss[:-1] > params.x_t)
        solids_flux = jnp.where(thickened, limited, flux[1:])

        source = state[1:] * self.schema.particulate_mask
        share = source / jnp.where(tss[1:] > 0.0, tss[1:], 1.0)[:, None]
        carried = solids_flux[:, None] * share
        empty = jnp.zeros((1, state.shape[1]))
        return jnp.concatenate([carried, empty]) - jnp.concatenate([empty, carried])
