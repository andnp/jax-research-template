"""Well-mixed liquid inventory: stored coordinates, wet-domain checks and the amount and volume balance.

Inventories are concentration-basis amounts times m³ (mol for a mol/m³ carrier), flows are m³/h and
rates are per hour. A chart maps the full transported inventories `n` to the stored coordinates `y`.
"""

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


class InventoryChart(Protocol):
    """A static map between full physical inventories and the coordinates a unit stores."""

    def coordinates(self, amounts: jax.Array) -> jax.Array: ...

    def amounts(self, coordinates: jax.Array) -> jax.Array:
        """Physical inventories from coordinates; an unreconstructable chart state is rejected, never clipped."""
        ...

    def coordinate_rate(self, amount_rate: jax.Array) -> jax.Array:
        """Stored-coordinate rate from the physical non-fast amount rate."""
        ...

    def reconstructable(self, coordinates: jax.Array) -> jax.Array:
        """Whether `amounts` accepts these coordinates, under the same roundoff tolerance; never raises."""
        ...


class IdentityChart:
    """Store the full transported inventories."""

    def coordinates(self, amounts: jax.Array) -> jax.Array:
        return amounts

    def amounts(self, coordinates: jax.Array) -> jax.Array:
        return coordinates

    def coordinate_rate(self, amount_rate: jax.Array) -> jax.Array:
        return amount_rate

    def reconstructable(self, coordinates: jax.Array) -> jax.Array:
        return jnp.all(jnp.isfinite(coordinates))


@jax_dataclass
class HoldupState:
    """Wet liquid volume (m³) and the chart's amount coordinates."""

    volume_m3: jax.Array
    amount_coordinates: jax.Array


class Inventory(NamedTuple):
    """Physical inventories and their concentrations at one wet state."""

    amounts: jax.Array
    concentrations: jax.Array


class InventoryBalance(NamedTuple):
    """Rates of one holdup: volume (m³/h), stored coordinates, and the physical non-fast amount flow."""

    volume_rate: jax.Array
    coordinate_rate: jax.Array
    amount_flow: jax.Array
    net_flow: jax.Array
    outflow: jax.Array


def reject_unless(valid: jax.Array, values: jax.Array, message: str) -> jax.Array:
    """`values` where `valid`; otherwise raise when evaluated eagerly, or return NaN when traced.

    A traced evaluation cannot raise, so an invalid stage poisons its result instead of looking like a
    valid zero.
    """
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(message)
    return jnp.where(valid, values, jnp.nan)


def reconstruct(chart: InventoryChart, state: HoldupState) -> Inventory:
    """Reconstruct inventories and concentrations, rejecting a nonpositive or nonfinite volume before dividing by it.

    The wet operating boundary `V >= V_min` is not checked here: it is a declared domain of the holdup's
    integration policy, located as an event, and smooth trials slightly beyond it stay evaluable.
    """
    volume = state.volume_m3
    volume = reject_unless(jnp.isfinite(volume) & (volume > 0), volume, "holdup volume must be finite and positive")
    amounts = chart.amounts(state.amount_coordinates)
    amounts = reject_unless(jnp.all(jnp.isfinite(amounts)), amounts, "holdup inventories must be finite")
    return Inventory(amounts, amounts / volume)


def inventory_balance(
    chart: InventoryChart,
    state: HoldupState,
    inventory: Inventory,
    inlet_flows: jax.Array,
    inlet_concentrations: jax.Array,
    draw_flows: jax.Array,
    overflow_flow: jax.Array,
    concentration_source: jax.Array,
    volume_source_flow: jax.Array | float = 0.0,
    amount_source: jax.Array | float = 0.0,
) -> InventoryBalance:
    """Volume and amount rates of a well-mixed holdup whose outlets all carry its concentration.

    `inlet_flows` has shape (inlets,) and `inlet_concentrations` (inlets, channels). The concentration
    source is a rate per m³ per hour and is multiplied by the volume here, once; `amount_source` is
    already an amount rate.
    """
    outflow = jnp.sum(draw_flows) + overflow_flow
    net_flow = jnp.sum(inlet_flows) - jnp.sum(draw_flows) + volume_source_flow
    amount_flow = (
        inlet_flows @ inlet_concentrations
        - outflow * inventory.concentrations
        + state.volume_m3 * concentration_source
        + amount_source
    )
    return InventoryBalance(net_flow - overflow_flow, chart.coordinate_rate(amount_flow), amount_flow, net_flow, outflow)
