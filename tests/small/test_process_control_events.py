import dataclasses
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import pytest
from process_control.event_location import (
    LocatorStatus,
    Trial,
    locate_crossing,
    run_zero_time_chain,
    select_earliest,
)
from process_control.event_registry import compile_registry
from process_control.events import Continuation, CrossingDirection, EventPoint, EventSpec, EventTolerance
from process_control.fields import LocalField
from process_control.integration_policy import IntegrationPolicy, legacy_policy
from process_control.integration_records import IntegrationContext

TOLERANCE = EventTolerance(time_h=1e-4, surface=1e-3)
ITERATIONS = 30
H_MAX = jnp.array(1.0)


def _affine(intercept: float | jax.Array, slope: float, admissible_up_to: float = float("inf")) -> Callable[[jax.Array], Trial]:
    def trial(h: jax.Array) -> Trial:
        return Trial(intercept + slope * h, h <= admissible_up_to)

    return trial


def _locate(
    trial: Callable[[jax.Array], Trial],
    start: float,
    direction: CrossingDirection = CrossingDirection.DOWN,
    iterations: int = ITERATIONS,
    tolerance: EventTolerance = TOLERANCE,
):
    return locate_crossing(trial, jnp.array(start), H_MAX, direction, tolerance, iterations)


class TestLocateCrossing:
    def test_linear_crossing_is_isolated_within_both_tolerances(self) -> None:
        """A surface falling through zero at h = 0.5 is bracketed narrowly with a small surface residual."""
        bracket = _locate(_affine(1.0, -2.0), 1.0)

        assert int(bracket.status) == LocatorStatus.LOCATED
        assert not bool(bracket.zero_time)
        assert float(bracket.upper_h - bracket.lower_h) <= TOLERANCE.time_h
        assert float(bracket.residual) <= TOLERANCE.surface
        assert float(bracket.event_h) == pytest.approx(0.5, abs=TOLERANCE.surface)

    def test_tighter_tolerance_moves_the_event_time_closer(self) -> None:
        """Event time error shrinks with the declared tolerance."""
        coarse = _locate(_affine(1.0, -3.0), 1.0, tolerance=EventTolerance(1e-1, 1e-1))
        fine = _locate(_affine(1.0, -3.0), 1.0)

        exact = 1.0 / 3.0
        assert abs(float(fine.event_h) - exact) < abs(float(coarse.event_h) - exact)

    def test_start_on_surface_departing_in_the_event_direction_fires_at_zero_time(self) -> None:
        """A downward event whose surface starts at zero and goes negative fires immediately."""
        bracket = _locate(_affine(0.0, -1.0), 0.0)

        assert int(bracket.status) == LocatorStatus.LOCATED
        assert bool(bracket.zero_time)
        assert float(bracket.event_h) == 0.0

    def test_start_on_surface_departing_inward_does_not_fire(self) -> None:
        """A downward event whose surface starts at zero and goes positive is not a crossing."""
        bracket = _locate(_affine(0.0, 1.0), 0.0)

        assert int(bracket.status) == LocatorStatus.NO_CROSSING
        assert not bool(bracket.zero_time)

    @pytest.mark.parametrize(
        ("direction", "slope", "fires"),
        [
            (CrossingDirection.UP, 1.0, True),
            (CrossingDirection.UP, -1.0, False),
            (CrossingDirection.BOTH, 1.0, True),
            (CrossingDirection.BOTH, -1.0, True),
        ],
    )
    def test_start_on_surface_follows_the_declared_direction(self, direction: CrossingDirection, slope: float, fires: bool) -> None:
        """Only an outgoing departure matching the declared direction fires at zero time."""
        bracket = _locate(_affine(0.0, slope), 0.0, direction)

        assert bool(bracket.zero_time) == fires

    def test_surface_that_never_reaches_zero_has_no_crossing(self) -> None:
        """A surface that stays positive over the interval reports no event."""
        assert int(_locate(_affine(1.0, -0.5), 1.0).status) == LocatorStatus.NO_CROSSING

    def test_crossing_against_the_declared_direction_is_ignored(self) -> None:
        """An upward event does not fire on a falling surface, nor a downward one on a rising surface."""
        assert int(_locate(_affine(1.0, -2.0), 1.0, CrossingDirection.UP).status) == LocatorStatus.NO_CROSSING
        assert int(_locate(_affine(-1.0, 2.0), -1.0, CrossingDirection.DOWN).status) == LocatorStatus.NO_CROSSING
        assert int(_locate(_affine(-1.0, 2.0), -1.0, CrossingDirection.UP).status) == LocatorStatus.LOCATED

    def test_inadmissible_upper_end_is_reduced_to_an_admissible_bracket(self) -> None:
        """When the full step leaves the valid domain, an admissible shorter trial can still prove the crossing."""
        bracket = _locate(_affine(0.2, -1.0, admissible_up_to=0.4), 0.2)

        assert int(bracket.status) == LocatorStatus.LOCATED
        assert float(bracket.event_h) == pytest.approx(0.2, abs=TOLERANCE.surface)

    def test_crossing_hidden_behind_an_inadmissible_region_is_a_bracket_failure(self) -> None:
        """An admissible shorter trial without a sign change cannot show there is no crossing further out."""
        bracket = _locate(_affine(0.6, -1.0, admissible_up_to=0.4), 0.6)

        assert int(bracket.status) == LocatorStatus.BRACKET_FAILED

    def test_inadmissible_interior_trial_is_a_bracket_failure(self) -> None:
        """A sign sample taken outside the valid domain is never used to move the bracket."""

        def trial(h: jax.Array) -> Trial:
            return Trial(0.5 - h, (h < 0.4) | (h > 0.6))

        assert int(_locate(trial, 0.5).status) == LocatorStatus.BRACKET_FAILED

    def test_exhausted_iteration_budget_is_not_converged(self) -> None:
        """Too few bisections leave the bracket wide, which is reported rather than accepted."""
        assert int(_locate(_affine(1.0, -3.0), 1.0, iterations=3).status) == LocatorStatus.NOT_CONVERGED

    def test_lanes_locate_independently_under_jit_and_vmap(self) -> None:
        """Each lane reports its own status and event time, matching a separate scalar call."""
        crossings = jnp.array([0.25, 0.5, 2.0])

        def lane(crossing: jax.Array) -> tuple[jax.Array, jax.Array]:
            bracket = locate_crossing(_affine(crossing, -1.0), crossing, H_MAX, CrossingDirection.DOWN, TOLERANCE, ITERATIONS)
            return bracket.status, bracket.event_h

        status, event_h = jax.jit(jax.vmap(lane))(crossings)

        assert status.tolist() == [LocatorStatus.LOCATED, LocatorStatus.LOCATED, LocatorStatus.NO_CROSSING]
        assert jnp.allclose(event_h[:2], crossings[:2], atol=TOLERANCE.surface)
        for index in range(3):
            scalar_status, scalar_h = lane(crossings[index])
            assert int(scalar_status) == int(status[index])
            assert float(scalar_h) == float(event_h[index])


class TestSelectEarliest:
    def test_earliest_found_event_wins(self) -> None:
        """Unfound events are ignored however early their placeholder time."""
        selection = select_earliest(jnp.array([True, False, True]), jnp.array([0.5, 0.0, 0.2]), 1e-6)

        assert int(selection.index) == 2
        assert selection.simultaneous.tolist() == [False, False, True]

    def test_events_within_the_tie_tolerance_are_simultaneous_and_registry_order_picks(self) -> None:
        """Compatible events inside the tolerance are all reported; the first in registry order is chosen."""
        selection = select_earliest(jnp.array([True, True, True]), jnp.array([0.30001, 0.3, 0.9]), 1e-3)

        assert selection.simultaneous.tolist() == [True, True, False]
        assert int(selection.index) == 0

    def test_nothing_found_selects_nothing(self) -> None:
        """No located event leaves the selection empty, including with no events registered."""
        assert not bool(select_earliest(jnp.array([False, False]), jnp.array([0.1, 0.2]), 1e-3).any)
        assert not bool(select_earliest(jnp.zeros((0,), bool), jnp.zeros((0,)), 1e-3).any)


def _event(
    name: str,
    sets: Mapping[str, int],
    fires_when: Callable[[Mapping[str, jax.Array]], jax.Array],
    direction: CrossingDirection = CrossingDirection.DOWN,
    continuation: Continuation = Continuation.TRANSITION,
) -> EventSpec[jax.Array, float]:
    def surface(state: jax.Array, params: float, modes: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array], fields: Mapping[LocalField, jax.Array]) -> jax.Array:
        return params - state

    def transition(modes: Mapping[str, jax.Array]) -> Mapping[str, jax.Array]:
        return {mode: jnp.array(value) for mode, value in sets.items()}

    return EventSpec(name, surface, "m3", direction, fires_when, TOLERANCE, transition, tuple(sets), continuation=continuation)


def _policy(modes: Mapping[str, str], *events: EventSpec[jax.Array, float]) -> IntegrationPolicy[jax.Array, float]:
    return IntegrationPolicy((), legacy_policy().check_state, modes=modes, events=events)


def _inactive(modes: Mapping[str, jax.Array]) -> jax.Array:
    return modes["cap"] == 0


def _context(**modes: int) -> IntegrationContext:
    return IntegrationContext({key.replace("_", "."): jnp.array(value, dtype=jnp.int32) for key, value in modes.items()})


class TestEventRegistry:
    def test_events_are_ordered_by_node_then_declaration(self) -> None:
        """Registry order is deterministic: node order, then each policy's declared order."""
        registry = compile_registry(
            {
                "tank": _policy({"cap": "p"}, _event("enter", {"cap": 1}, _inactive), _event("leave", {"cap": 0}, _inactive)),
                "sump": _policy({"cap": "q"}, _event("enter", {"cap": 1}, _inactive)),
            }
        )

        assert registry.keys == (("tank", "enter"), ("tank", "leave"), ("sump", "enter"))

    def test_surfaces_and_activity_read_each_nodes_own_state_and_modes(self) -> None:
        """Each event surface sees its node's state, parameters and local modes."""
        registry = compile_registry({"tank": _policy({"cap": "p"}, _event("enter", {"cap": 1}, _inactive))})
        point = EventPoint(
            state={"tank": jnp.array(4.0)},
            params={"tank": 10.0},
            context=_context(tank_cap=0),
            inputs={"tank": {}},
            fields={"tank": {}},
        )

        assert registry.surfaces(point).tolist() == [6.0]
        assert registry.active(point.context).tolist() == [True]
        assert registry.active(_context(tank_cap=1)).tolist() == [False]

    def test_transition_sets_only_its_declared_modes_and_keeps_dtypes(self) -> None:
        """Applying an event changes its own modes in the context's dtype and leaves the others alone."""
        registry = compile_registry(
            {"tank": _policy({"cap": "p", "other": "q"}, _event("enter", {"cap": 1}, _inactive))}
        )
        context = _context(tank_cap=0, tank_other=7)

        updated = registry.apply(jnp.array(0), context)

        assert int(updated.modes["tank.cap"]) == 1
        assert int(updated.modes["tank.other"]) == 7
        assert updated.modes["tank.cap"].dtype == jnp.int32

    def test_transition_that_sets_other_modes_than_declared_is_rejected(self) -> None:
        """A transition writing a mode it did not declare would break conflict detection."""
        liar = dataclasses.replace(_event("enter", {"cap": 1}, _inactive), sets_modes=("other",))
        registry = compile_registry({"tank": _policy({"cap": "p", "other": "q"}, liar)})

        with pytest.raises(ValueError, match="must set exactly"):
            registry.apply(jnp.array(0), _context(tank_cap=0, tank_other=0))

    def test_events_writing_the_same_mode_conflict_only_when_both_selected(self) -> None:
        """Simultaneous events with a shared written mode are incompatible; disjoint ones are not."""
        registry = compile_registry(
            {
                "tank": _policy(
                    {"cap": "p", "other": "q"},
                    _event("a", {"cap": 1}, _inactive),
                    _event("b", {"cap": 0}, _inactive),
                    _event("c", {"other": 1}, _inactive),
                )
            }
        )

        assert bool(registry.conflicts(jnp.array([True, True, False])))
        assert not bool(registry.conflicts(jnp.array([True, False, True])))
        assert not bool(registry.conflicts(jnp.array([True, False, False])))

    def test_policy_rejects_events_that_set_undeclared_modes(self) -> None:
        """Events may only write modes the policy declares."""
        with pytest.raises(ValueError, match="sets undeclared modes"):
            _policy({"cap": "p"}, _event("enter", {"missing": 1}, _inactive))

    def test_policy_rejects_duplicate_event_names(self) -> None:
        """Events are keyed by (node, name), so a node cannot reuse a name."""
        with pytest.raises(ValueError, match="event names must be unique"):
            _policy({"cap": "p"}, _event("enter", {"cap": 1}, _inactive), _event("enter", {"cap": 0}, _inactive))


class TestZeroTimeChain:
    def test_event_that_disarms_itself_fires_once(self) -> None:
        """A start-on-surface event moves its mode, which deactivates it, so it does not fire again."""
        registry = compile_registry({"tank": _policy({"cap": "p"}, _event("enter", {"cap": 1}, _inactive))})

        def fires(context: IntegrationContext) -> jax.Array:
            return jnp.array([True])

        result = run_zero_time_chain(registry, _context(tank_cap=0), fires, max_transitions=4)

        assert int(result.transitions) == 1
        assert int(result.context.modes["tank.cap"]) == 1
        assert not bool(result.exhausted)
        assert not bool(result.incompatible)

    def test_events_that_keep_rearming_each_other_exhaust_the_budget(self) -> None:
        """Mutually re-enabling transitions are flagged as chatter instead of looping."""

        def always(modes: Mapping[str, jax.Array]) -> jax.Array:
            return jnp.array(True)

        def surface(state: jax.Array, params: float, modes: Mapping[str, jax.Array], inputs: Mapping[str, jax.Array], fields: Mapping[LocalField, jax.Array]) -> jax.Array:
            return params - state

        def flip(modes: Mapping[str, jax.Array]) -> Mapping[str, jax.Array]:
            return {"cap": 1 - modes["cap"]}

        toggler = EventSpec("toggle", surface, "m3", CrossingDirection.DOWN, always, TOLERANCE, flip, ("cap",))
        registry = compile_registry({"tank": _policy({"cap": "p"}, toggler)})

        def fires(context: IntegrationContext) -> jax.Array:
            return jnp.array([True])

        result = run_zero_time_chain(registry, _context(tank_cap=0), fires, max_transitions=3)

        assert int(result.transitions) == 3
        assert bool(result.exhausted)

    def test_simultaneous_firing_events_with_clashing_transitions_halt_unapplied(self) -> None:
        """Two events writing the same mode cannot both be honoured, so nothing is applied and the clash is reported."""
        registry = compile_registry(
            {"tank": _policy({"cap": "p"}, _event("a", {"cap": 1}, _inactive), _event("b", {"cap": 2}, _inactive))}
        )

        def fires(context: IntegrationContext) -> jax.Array:
            return jnp.array([True, True])

        result = run_zero_time_chain(registry, _context(tank_cap=0), fires, max_transitions=4)

        assert bool(result.incompatible)
        assert int(result.transitions) == 0
        assert int(result.context.modes["tank.cap"]) == 0

    def test_empty_registry_leaves_the_context_alone(self) -> None:
        """A plant with no events needs no chain."""
        context = _context()

        def fires(context: IntegrationContext) -> jax.Array:
            return jnp.zeros((0,), bool)

        result = run_zero_time_chain(compile_registry({}), context, fires, max_transitions=2)

        assert int(result.transitions) == 0
        assert not bool(result.exhausted)
