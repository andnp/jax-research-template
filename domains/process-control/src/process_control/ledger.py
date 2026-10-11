from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.integration_records import IntegrationContext

type Rates = Mapping[str, jax.Array]


@dataclass(frozen=True)
class LedgerSpec[S, E]:
    """A named set of accepted-quadrature rates with their units, extracted purely from one RK stage.

    `rates` receives the stage state, its transient evaluation, the routed inputs and the context, and returns
    one array per name in `units`.
    """

    name: str
    units: Mapping[str, str]
    rates: Callable[[S, E, Mapping[str, jax.Array], IntegrationContext], Rates]

    def evaluate(self, state: S, evaluation: E, inputs: Mapping[str, jax.Array], context: IntegrationContext) -> dict[str, jax.Array]:
        rates = self.rates(state, evaluation, inputs, context)
        if set(rates) != set(self.units):
            raise ValueError(f"ledger {self.name!r} must return rates {sorted(self.units)}, got {sorted(rates)}")
        return {name: rates[name] for name in self.units}


def rk4_weighted_rate[T](k1: T, k2: T, k3: T, k4: T) -> T:
    """Combine four stage rate pytrees leaf by leaf in the RK4 expression order."""

    def combine(a: jax.Array, b: jax.Array, c: jax.Array, d: jax.Array) -> jax.Array:
        return (a + 2.0 * b + 2.0 * c + d) / 6.0

    return jax.tree.map(combine, k1, k2, k3, k4)


def segment_integral[T](k1: T, k2: T, k3: T, k4: T, duration: jax.Array) -> T:
    """Integrate stage rates over one segment: the weighted rate times the segment duration."""

    def scale(rate: jax.Array) -> jax.Array:
        return rate * duration

    return jax.tree.map(scale, rk4_weighted_rate(k1, k2, k3, k4))


def commit_if_accepted[T](total: T, segment: T, accepted: jax.Array) -> T:
    """Add a segment's quadratures to the running totals only when the segment was accepted."""

    def commit(running: jax.Array, increment: jax.Array) -> jax.Array:
        return jnp.where(accepted, running + increment, running)

    return jax.tree.map(commit, total, segment)


@jax_dataclass
class FastExtentLedger:
    """Reaction extent implied by an accepted segment, with the all-row balance check it must satisfy."""

    extent: jax.Array
    residual: jax.Array
    rows_balanced: jax.Array
    progress_nonnegative: jax.Array

    @property
    def ok(self) -> jax.Array:
        return self.rows_balanced & self.progress_nonnegative


def fast_extent_ledger(
    n_start: jax.Array,
    n_end: jax.Array,
    nonfast_integral: jax.Array,
    stoichiometry: jax.Array,
    row: int,
    rtol: float,
    atol: float,
) -> FastExtentLedger:
    """Recover the one-column fast extent from a reconstructed chart row and verify every inventory row.

    With `delta_n = n_end - n_start` and the integrated nonfast rates, the fast reaction must explain the rest:
    `delta_n - nonfast_integral = stoichiometry * extent`. The extent comes from `row`, the chart's reference
    species, and is never applied to the state again. Irreversible progress must not be negative beyond `atol`.
    """
    delta_n = n_end - n_start
    extent = (delta_n[row] - nonfast_integral[row]) / stoichiometry[row]
    explained = stoichiometry * extent
    residual = delta_n - nonfast_integral - explained
    scale = jnp.maximum(jnp.maximum(jnp.abs(delta_n), jnp.abs(nonfast_integral)), jnp.abs(explained))
    return FastExtentLedger(
        extent=extent,
        residual=residual,
        rows_balanced=jnp.all(jnp.abs(residual) <= atol + rtol * scale),
        progress_nonnegative=extent >= -atol,
    )
