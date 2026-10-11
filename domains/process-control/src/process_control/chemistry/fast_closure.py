"""Fixed-yield fast reaction closure: an analytic one-progress invariant chart over a transported carrier.

Two reagents, a limiting reactant A (one unit of extent consumes one unit of it) and a second reagent B
consumed `d` per unit extent, react instantaneously to fixed products. The stored invariants are

    I   = n_B - d n_A                           (signed; positive when B is in excess)
    J_j = n_j + a_j n_A + b_j n_B               (one per product channel j, with a_j + d b_j = S_j)

and the physical inventories follow from `n_A = max(-I/d, 0)`, `n_B = max(I, 0)`. The `max` is the
reagent-exhaustion chart itself, not a clip: reconstructed products must come out non-negative or the
chart state is rejected. Channels the reaction does not touch stay identity coordinates, ordered first.
The yield is fixed for a trajectory; changing it changes the chart.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from process_control.binding import Binding, NamedAccess, Requirement
from process_control.chemistry.basis import AqueousBasis, compile_basis_stoichiometry
from process_control.chemistry.network import row_component_basis
from process_control.chemistry.reactions import ReactionSet
from process_control.components import ComponentSchema
from process_control.inventory import reject_unless

_TOLERANCE = 1e-12


class FastInitialization(NamedTuple):
    """Initial coordinates, the relaxed physical inventories and the one-time extent (amount of reactant A)."""

    coordinates: jax.Array
    amounts: jax.Array
    extent: jax.Array


@dataclass(frozen=True, eq=False)
class FastReactionClosure:
    """A validated fixed-yield closure and its `InventoryChart`.

    `stoichiometry` is the carrier-length change per unit extent, with reactant A at -1 and reagent B at
    -d. Every positive entry is a product, taking its stoichiometry from A unless it is in
    `oxidant_products`, which take it from B. Build with `make_fixed_yield_closure`.
    """

    schema: ComponentSchema
    stoichiometry: np.ndarray
    reactant: int
    oxidant: int
    oxidant_products: tuple[int, ...]
    untouched: tuple[int, ...] = field(init=False)
    products: tuple[int, ...] = field(init=False)
    demand: float = field(init=False)
    invariants: np.ndarray = field(init=False)
    _from_reactant: np.ndarray = field(init=False)
    _from_oxidant: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        s, a, b = self.stoichiometry, self.reactant, self.oxidant
        if s.shape != (len(self.schema.names),):
            raise ValueError(f"stoichiometry has shape {s.shape} for a carrier of {len(self.schema.names)} channels")
        if a == b or not np.isclose(s[a], -1.0) or not s[b] < 0:
            raise ValueError("the reactant must have coefficient -1 and the oxidant a negative coefficient")
        if any(s[i] < 0 for i in range(len(s)) if i not in (a, b)):
            raise ValueError("only the reactant and the oxidant may be consumed")

        products = tuple(i for i in range(len(s)) if s[i] > 0)
        if not products:
            raise ValueError("a fast reaction needs at least one product")
        if not set(self.oxidant_products) <= set(products):
            raise ValueError("oxidant products must be product channels")
        untouched = tuple(i for i in range(len(s)) if s[i] == 0)

        d = float(-s[b])
        from_oxidant = np.array([s[j] / d if j in self.oxidant_products else 0.0 for j in products])
        from_reactant = np.array([0.0 if j in self.oxidant_products else s[j] for j in products])
        rows = np.zeros((len(untouched) + 1 + len(products), len(s)))
        for row, i in enumerate(untouched):
            rows[row, i] = 1.0
        rows[len(untouched), b], rows[len(untouched), a] = 1.0, -d
        for row, (j, from_a, from_b) in enumerate(zip(products, from_reactant, from_oxidant, strict=True), len(untouched) + 1):
            rows[row, j], rows[row, a], rows[row, b] = 1.0, from_a, from_b

        if np.abs(rows @ s).max() > _TOLERANCE:
            raise ValueError("the invariant chart does not annihilate the fast stoichiometry (U S != 0)")
        if np.linalg.matrix_rank(rows) != len(s) - 1:
            raise ValueError("the invariant chart must have rank one less than the carrier")

        for name, value in (
            ("untouched", untouched),
            ("products", products),
            ("demand", d),
            ("invariants", rows),
            ("_from_reactant", from_reactant),
            ("_from_oxidant", from_oxidant),
        ):
            object.__setattr__(self, name, value)

    def coordinates(self, amounts: jax.Array) -> jax.Array:
        return jnp.asarray(self.invariants, dtype=amounts.dtype) @ amounts

    def amounts(self, coordinates: jax.Array) -> jax.Array:
        full, admissible = self._reconstruct(coordinates)
        return reject_unless(admissible, full, "reconstructed fast-closure products must be non-negative")

    def reconstructable(self, coordinates: jax.Array) -> jax.Array:
        return jnp.all(jnp.isfinite(coordinates)) & self._reconstruct(coordinates)[1]

    def _reconstruct(self, coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Physical inventories, and whether every product is non-negative up to the roundoff of the subtraction that forms it."""
        nu = len(self.untouched)
        invariant = coordinates[nu]
        reactant = jnp.maximum(-invariant / self.demand, 0.0)
        oxidant = jnp.maximum(invariant, 0.0)
        dtype = coordinates.dtype
        from_reactant = jnp.asarray(self._from_reactant, dtype) * reactant
        from_oxidant = jnp.asarray(self._from_oxidant, dtype) * oxidant
        products = coordinates[nu + 1 :] - from_reactant - from_oxidant
        scale = jnp.abs(coordinates[nu + 1 :]) + jnp.abs(from_reactant) + jnp.abs(from_oxidant)

        full = jnp.zeros(len(self.schema.names), dtype=dtype)
        full = full.at[jnp.array(self.untouched, dtype=int)].set(coordinates[:nu])
        full = full.at[self.reactant].set(reactant).at[self.oxidant].set(oxidant)
        full = full.at[jnp.array(self.products, dtype=int)].set(products)
        return full, jnp.all(products >= -_TOLERANCE * scale)

    def coordinate_rate(self, amount_rate: jax.Array) -> jax.Array:
        return jnp.asarray(self.invariants, dtype=amount_rate.dtype) @ amount_rate

    def initialize(self, amounts: jax.Array) -> FastInitialization:
        """React coexisting reagents once and return the invariant coordinates of the relaxed inventories."""
        relaxed, extent = self.transform_amounts(amounts)
        return FastInitialization(self.coordinates(relaxed), relaxed, extent)

    def transform_amounts(self, amounts: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Relax non-negative amounts, or amount flows per hour, to the closure's endpoint; returns them with the extent.

        The extent has the units of the input: mol for inventories, mol/h for flows.
        """
        valid = jnp.all(jnp.isfinite(amounts) & (amounts >= 0.0))
        amounts = reject_unless(valid, amounts, "fast-closure amounts must be finite and non-negative")
        extent = jnp.minimum(amounts[self.reactant], amounts[self.oxidant] / self.demand)
        return amounts + jnp.asarray(self.stoichiometry, dtype=amounts.dtype) * extent, extent

    def extent_rate(self, coordinates: jax.Array, amount_rate: jax.Array) -> jax.Array:
        """Directional rate of the fast extent (mol/h) from the physical non-fast amount rate.

        Positive reagent supply is consumed on the excess side, and the minimum of the two supplies when
        both are absent. Negative supply of an absent reagent is rejected as an invalid source law.
        """
        invariant = coordinates[len(self.untouched)]
        supply_reactant = amount_rate[self.reactant]
        supply_oxidant = amount_rate[self.oxidant] / self.demand
        rate = jnp.where(invariant > 0, supply_reactant, jnp.where(invariant < 0, supply_oxidant, jnp.minimum(supply_reactant, supply_oxidant)))
        admissible = jnp.where(
            invariant > 0,
            supply_reactant >= 0,
            jnp.where(invariant < 0, supply_oxidant >= 0, (supply_reactant >= 0) & (supply_oxidant >= 0)),
        )
        return reject_unless(admissible, rate, "an absent reagent cannot have negative supply")


def make_fixed_yield_closure(
    basis: AqueousBasis,
    reactions: ReactionSet,
    yields: Mapping[str, float],
    reactant: str,
    oxidant: str,
    oxidant_products: Sequence[str],
    schema: ComponentSchema,
    channel_map: Mapping[str, str] | None = None,
) -> FastReactionClosure:
    """Mix `reactions` by their yields into one fast stoichiometry and bind it to the carrier `schema`.

    `yields` maps every reaction name to its share (finite, in [0, 1], summing to one). Names of the
    reactant, oxidant and oxidant products are basis row names; `channel_map` renames them to carrier
    channels, whose bases must match the rows' exactly.
    """
    names = [reaction.name for reaction in reactions.reactions]
    if set(yields) != set(names):
        raise ValueError(f"yields must name exactly the reactions {names}, got {sorted(yields)}")
    weights = np.array([yields[name] for name in names], dtype=np.float64)
    if not np.all(np.isfinite(weights) & (weights >= 0.0) & (weights <= 1.0)) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"yields must lie in [0, 1] and sum to one, got {yields}")

    local = compile_basis_stoichiometry(basis, reactions) @ weights
    changed = [i for i in range(len(basis.rows)) if local[i] != 0.0]
    access = NamedAccess(writes=tuple(Requirement(basis.names[i], row_component_basis(basis, basis.rows[i])) for i in changed))
    binding = Binding.resolve(access, schema, channel_map)

    stoichiometry = np.zeros(len(schema.names))
    stoichiometry[list(binding.write_index)] = local[changed]
    index = {basis.names[i]: binding.write_index[k] for k, i in enumerate(changed)}
    unknown = [name for name in (reactant, oxidant, *oxidant_products) if name not in index]
    if unknown:
        raise ValueError(f"{unknown} are not changed by the fast reactions")
    return FastReactionClosure(schema, stoichiometry, index[reactant], index[oxidant], tuple(index[name] for name in oxidant_products))
