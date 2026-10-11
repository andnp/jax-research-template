from collections.abc import Callable
from enum import IntEnum

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.event_registry import EventRegistry
from process_control.events import CrossingDirection, EventTolerance
from process_control.integration_records import IntegrationContext


class LocatorStatus(IntEnum):
    NO_CROSSING = 0
    LOCATED = 1
    BRACKET_FAILED = 2
    NOT_CONVERGED = 3


@jax_dataclass
class Trial:
    """The surface value at the end of one complete trial, and whether that trial stayed in the valid domain."""

    surface: jax.Array
    admissible: jax.Array


@jax_dataclass
class Bracket:
    """Where a crossing was isolated: the bracket, its surface values, and the chosen near-boundary endpoint."""

    status: jax.Array
    lower_h: jax.Array
    upper_h: jax.Array
    lower_surface: jax.Array
    upper_surface: jax.Array
    event_h: jax.Array
    residual: jax.Array
    zero_time: jax.Array
    iterations: jax.Array


@jax_dataclass
class EventSelection:
    """The earliest located event, with every event within the tie tolerance of it."""

    any: jax.Array
    index: jax.Array
    simultaneous: jax.Array


@jax_dataclass
class ChainResult:
    """Context after the zero-time transition chain, and why the chain stopped if it did not settle."""

    context: IntegrationContext
    transitions: jax.Array
    exhausted: jax.Array
    incompatible: jax.Array


def locate_crossing(
    trial: Callable[[jax.Array], Trial],
    surface_start: jax.Array,
    h_max: jax.Array,
    direction: CrossingDirection,
    tolerance: EventTolerance,
    max_iterations: int,
) -> Bracket:
    """Isolate a directional surface crossing within (0, h_max] by bisection over complete trials.

    `trial(h)` must run the full step of length `h` from the accepted origin with the pre-event modes held
    fixed. A start on the surface fires at zero time only when the surface departs in the declared direction,
    so an event that has just fired is not found again. An inadmissible upper end is reduced by halving until
    an admissible sample exists; a crossing is only claimed when an admissible sample proves it, and a reduced
    interval without a sign change is a bracket failure rather than evidence of no crossing.
    """
    upper_h, upper_g, has_upper = _admissible_upper(trial, h_max, max_iterations)
    reduced = upper_h < h_max
    on_surface = jnp.abs(surface_start) <= tolerance.surface
    armed = _armed(direction, surface_start)
    departs = _departs(direction, upper_g, tolerance.surface)
    crosses = ~on_surface & armed & has_upper & (surface_start * upper_g <= 0.0)
    zero_time = on_surface & has_upper & departs
    side = jnp.sign(surface_start)

    def settled(lower_h: jax.Array, upper_h: jax.Array, lower_g: jax.Array, upper_g: jax.Array) -> jax.Array:
        narrow = upper_h - lower_h <= tolerance.time_h
        return narrow & (jnp.minimum(jnp.abs(lower_g), jnp.abs(upper_g)) <= tolerance.surface)

    def refine(_: int, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        lower_h, upper_h, lower_g, upper_g, failed, iterations = carry
        active = crosses & ~failed & ~settled(lower_h, upper_h, lower_g, upper_g)

        def bisect(_: None) -> tuple[jax.Array, ...]:
            mid = lower_h + 0.5 * (upper_h - lower_h)
            sample = trial(mid)
            move_lower = sample.admissible & (sample.surface * side > 0.0)
            move_upper = sample.admissible & ~move_lower
            return (
                jnp.where(move_lower, mid, lower_h),
                jnp.where(move_upper, mid, upper_h),
                jnp.where(move_lower, sample.surface, lower_g),
                jnp.where(move_upper, sample.surface, upper_g),
                ~sample.admissible,
                iterations + 1,
            )

        def hold(_: None) -> tuple[jax.Array, ...]:
            return lower_h, upper_h, lower_g, upper_g, failed, iterations

        return jax.lax.cond(active, bisect, hold, None)

    start = (jnp.zeros(()), upper_h, surface_start, upper_g, jnp.zeros((), bool), jnp.zeros((), jnp.int32))
    lower_h, upper_h, lower_g, upper_g, failed, iterations = jax.lax.fori_loop(0, max_iterations, refine, start)

    status = jnp.select(
        [~has_upper, on_surface & departs, on_surface, ~armed, crosses & failed, crosses & settled(lower_h, upper_h, lower_g, upper_g), crosses, reduced],
        [
            LocatorStatus.BRACKET_FAILED,
            LocatorStatus.LOCATED,
            LocatorStatus.NO_CROSSING,
            LocatorStatus.NO_CROSSING,
            LocatorStatus.BRACKET_FAILED,
            LocatorStatus.LOCATED,
            LocatorStatus.NOT_CONVERGED,
            LocatorStatus.BRACKET_FAILED,
        ],
        LocatorStatus.NO_CROSSING,
    )
    residual = jnp.minimum(jnp.abs(lower_g), jnp.abs(upper_g))
    near_lower = jnp.abs(lower_g) <= jnp.abs(upper_g)
    return Bracket(
        status=status.astype(jnp.int32),
        lower_h=lower_h,
        upper_h=upper_h,
        lower_surface=lower_g,
        upper_surface=upper_g,
        event_h=jnp.where(zero_time, 0.0, jnp.where(near_lower, lower_h, upper_h)),
        residual=jnp.where(zero_time, jnp.abs(surface_start), residual),
        zero_time=zero_time,
        iterations=iterations,
    )


def _admissible_upper(trial: Callable[[jax.Array], Trial], h_max: jax.Array, max_halvings: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The longest of h_max, h_max / 2, ... whose trial is admissible, with its surface value and whether one exists."""

    def halve(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        h, g, found = carry

        def probe(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
            candidate = h_max * 0.5**i
            sample = trial(candidate)
            return candidate, sample.surface, sample.admissible

        def keep(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
            return carry

        return jax.lax.cond(found, keep, probe, None)

    start = (h_max, jnp.zeros(()), jnp.zeros((), bool))
    return jax.lax.fori_loop(0, max_halvings + 1, halve, start)


def _armed(direction: CrossingDirection, surface_start: jax.Array) -> jax.Array:
    match direction:
        case CrossingDirection.DOWN:
            return surface_start > 0.0
        case CrossingDirection.UP:
            return surface_start < 0.0
        case CrossingDirection.BOTH:
            return jnp.ones((), bool)


def _departs(direction: CrossingDirection, surface: jax.Array, tolerance: float) -> jax.Array:
    match direction:
        case CrossingDirection.DOWN:
            return surface < -tolerance
        case CrossingDirection.UP:
            return surface > tolerance
        case CrossingDirection.BOTH:
            return jnp.abs(surface) > tolerance


def select_earliest(found: jax.Array, times: jax.Array, tie_h: float) -> EventSelection:
    """Pick the earliest located event; events within `tie_h` of it are simultaneous and registry order picks among them."""
    if found.shape[0] == 0:
        return EventSelection(jnp.zeros((), bool), jnp.zeros((), jnp.int32), found)
    masked = jnp.where(found, times, jnp.inf)
    simultaneous = found & (masked <= jnp.min(masked) + tie_h)
    return EventSelection(jnp.any(found), jnp.argmax(simultaneous), simultaneous)


def run_zero_time_chain(
    registry: EventRegistry, context: IntegrationContext, fires: Callable[[IntegrationContext], jax.Array], max_transitions: int
) -> ChainResult:
    """Apply transitions of events firing at the current instant until none fires, within a static budget.

    `fires(context)` says which events' surfaces sit on their boundary with an outgoing crossing. Running out
    of budget means the transitions keep re-arming each other; a firing set whose transitions write the same
    mode is incompatible and halts the chain unapplied.
    """
    if not registry.events:
        return ChainResult(context, jnp.zeros((), jnp.int32), jnp.zeros((), bool), jnp.zeros((), bool))

    def firing(current: IntegrationContext) -> jax.Array:
        return fires(current) & registry.active(current)

    def step(_: int, carry: tuple[IntegrationContext, jax.Array, jax.Array]) -> tuple[IntegrationContext, jax.Array, jax.Array]:
        current, count, incompatible = carry
        pending = firing(current)
        clash = incompatible | registry.conflicts(pending)
        go = jnp.any(pending) & ~clash

        def apply(c: IntegrationContext) -> IntegrationContext:
            return registry.apply(jnp.argmax(pending), c)

        def keep(c: IntegrationContext) -> IntegrationContext:
            return c

        advanced = jax.lax.cond(go, apply, keep, current)
        return advanced, count + go.astype(jnp.int32), clash

    start = (context, jnp.zeros((), jnp.int32), jnp.zeros((), bool))
    final, count, incompatible = jax.lax.fori_loop(0, max_transitions, step, start)
    return ChainResult(final, count, jnp.any(firing(final)) & ~incompatible, incompatible)
