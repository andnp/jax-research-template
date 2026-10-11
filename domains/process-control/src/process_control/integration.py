"""The one guarded explicit integrator: an RK4 trial kernel inside a bounded, transactional advance.

Every continuous model advances through `advance`. A trial runs four guarded stage evaluations of the
whole system with modes frozen; the driver alone decides acceptance, halves rejected durations, locates
mode-changing events, commits stage quadratures only with the accepted segment, and applies the
projection once per nominal substep. All loops have static bounds, so the advance stages under `jit`,
batches under `vmap` lane by lane, and differentiates through its scan lowering.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.event_location import Bracket, LocatorStatus, Trial, locate_crossing, run_zero_time_chain, select_earliest
from process_control.event_registry import EventRegistry, RegisteredEvent
from process_control.events import Continuation, CrossingDirection
from process_control.integration_records import NO_REASON, AdvanceChecks, AdvanceResult, AdvanceStatus, CheckResult, IntegrationContext
from process_control.ledger import commit_if_accepted, rk4_weighted_rate, segment_integral

PROJECTION = "projection"
NONFINITE_STATE = "finite_state"


@jax_dataclass
class StateCheck:
    """Whether a state fails a hard domain check or only a continuation check, and the first failing reason id."""

    hard: jax.Array
    continuation: jax.Array
    reason: jax.Array


@jax_dataclass
class StageResult[S]:
    """One guarded stage: the derivative, its ledger rates, and the pre- and post-evaluation checks combined."""

    derivative: S
    rates: Mapping[str, Mapping[str, jax.Array]]
    check: StateCheck


@jax_dataclass
class TrialResult[S]:
    """A complete RK4 trial: the raw endpoint, its integrated ledger increments and every stage and endpoint check.

    `origin_failed` marks a failure of the first stage, which no shorter duration can cure.
    """

    state: S
    increments: Mapping[str, Mapping[str, jax.Array]]
    check: StateCheck
    origin_failed: jax.Array


@dataclass(frozen=True)
class GuardedSystem[S]:
    """What the driver needs from a model: guarded stages, an endpoint check, the projection and its events.

    `stage` checks a state before evaluating it and validates the result; `check` only checks. `surfaces`
    evaluates every registered event surface at a state, and `observe_projection` records a quantity of
    each projection from its raw and projected endpoints.
    """

    stage: Callable[[S, IntegrationContext], StageResult[S]]
    check: Callable[[S, IntegrationContext], StateCheck]
    project: Callable[[S], S]
    registry: EventRegistry = EventRegistry(())
    surfaces: Callable[[S, IntegrationContext], jax.Array] | None = None
    observe_projection: Callable[[S, S], Mapping[str, jax.Array]] | None = None


@dataclass(frozen=True)
class AdvanceBudget:
    """Static work limits per nominal substep: trials (accepted or rejected), located events, bisection steps, zero-time transitions."""

    attempts: int = 16
    events: int = 4
    location_iterations: int = 64
    transitions: int = 4


DEFAULT_BUDGET = AdvanceBudget()


def state_check(result: CheckResult, continuation: tuple[bool, ...]) -> StateCheck:
    """Split ordered check flags into hard and continuation failures; `continuation` marks each flag."""
    mask = jnp.asarray(continuation, dtype=bool)
    return StateCheck(jnp.any(result.failed & ~mask), jnp.any(result.failed & mask), result.first_reason)


def rk4_trial[S](system: GuardedSystem[S], state: S, context: IntegrationContext, h: jax.Array) -> TrialResult[S]:
    """The sole RK4 kernel: four guarded stages from `state` over `h`, combined in the classical expression order."""
    e1 = system.stage(state, context)
    e2 = system.stage(_tree_add(state, _tree_scale(e1.derivative, h * 0.5)), context)
    e3 = system.stage(_tree_add(state, _tree_scale(e2.derivative, h * 0.5)), context)
    e4 = system.stage(_tree_add(state, _tree_scale(e3.derivative, h)), context)
    dy = rk4_weighted_rate(e1.derivative, e2.derivative, e3.derivative, e4.derivative)
    # The barrier keeps guard and bookkeeping consumers from changing how XLA fuses, and so rounds, the update.
    raw = jax.lax.optimization_barrier(_tree_add(state, _tree_scale(dy, h)))
    increments = segment_integral(e1.rates, e2.rates, e3.rates, e4.rates, h)
    checks = _first_failure((e1.check, e2.check, e3.check, e4.check, system.check(raw, context)))
    return TrialResult(raw, increments, checks, e1.check.hard)


def advance[S](
    system: GuardedSystem[S],
    state: S,
    context: IntegrationContext,
    dt: jax.Array | float,
    substeps: int,
    budget: AdvanceBudget = DEFAULT_BUDGET,
) -> AdvanceResult[S]:
    """Advance over `dt` in `substeps` nominal substeps, each `dt / substeps` unless a guard or an event splits it.

    The result holds the last accepted state, the accepted duration (exactly `dt` on success), the
    status, the accepted quadratures and the context to pass to the next call. On failure the state is
    accepted progress only, never a rejected trial, and its completed time says how far it got.
    """
    dt = jnp.asarray(dt)
    sub_dt = dt / substeps
    zero = jnp.zeros((), jnp.int32)

    initial = system.check(state, context)
    start = _Progress(
        state=state,
        context=context,
        quadratures=_zero_quadratures(system, state, context, sub_dt),
        status=jnp.where(initial.hard | initial.continuation, AdvanceStatus.INVALID_INITIAL_STATE, AdvanceStatus.SUCCESS).astype(jnp.int32),
        reason=initial.reason.astype(jnp.int32),
        event_index=jnp.full((), -1, jnp.int32),
        completed=jnp.zeros_like(sub_dt),
        retries=zero,
        segments=zero,
        location_iterations=zero,
        transitions=zero,
    )

    end = jax.lax.fori_loop(0, substeps, _nominal_substep(system, sub_dt, budget), start)
    succeeded = end.status == AdvanceStatus.SUCCESS
    return AdvanceResult(
        state=end.state,
        completed_time_h=jnp.where(succeeded, dt, end.completed),
        status=end.status,
        quadratures=end.quadratures,
        context=end.context,
        checks=AdvanceChecks(end.reason, end.event_index, end.retries, end.segments, end.location_iterations, end.transitions),
    )


def advance_ode[S](
    derivative: Callable[[S], S],
    project: Callable[[S], S],
    state: S,
    dt: jax.Array | float,
    substeps: int,
    observe_projection: Callable[[S, S], Mapping[str, jax.Array]] | None = None,
    budget: AdvanceBudget = DEFAULT_BUDGET,
) -> AdvanceResult[S]:
    """Advance a bare derivative through the same guarded driver; its only domain is a finite state.

    Stages are not checked one by one: a nonfinite stage poisons the trial endpoint, whose check rejects it.
    """

    def check(current: S, context: IntegrationContext) -> StateCheck:
        bad = _nonfinite(current)
        return StateCheck(bad, jnp.zeros((), bool), jnp.where(bad, 1, NO_REASON).astype(jnp.int32))

    def stage(current: S, context: IntegrationContext) -> StageResult[S]:
        unchecked = jnp.zeros((), bool)
        return StageResult(derivative(current), {}, StateCheck(unchecked, unchecked, jnp.asarray(NO_REASON, jnp.int32)))

    system = GuardedSystem(stage, check, project, observe_projection=observe_projection)
    return advance(system, state, IntegrationContext({}), dt, substeps, budget)


@jax_dataclass
class _Progress[S]:
    state: S
    context: IntegrationContext
    quadratures: Mapping[str, Mapping[str, jax.Array]]
    status: jax.Array
    reason: jax.Array
    event_index: jax.Array
    completed: jax.Array
    retries: jax.Array
    segments: jax.Array
    location_iterations: jax.Array
    transitions: jax.Array


@jax_dataclass
class _Segment[S]:
    state: S
    context: IntegrationContext
    quadratures: Mapping[str, Mapping[str, jax.Array]]
    remaining: jax.Array
    h_try: jax.Array
    accepted: jax.Array
    done: jax.Array
    status: jax.Array
    reason: jax.Array
    event_index: jax.Array
    events: jax.Array
    retries: jax.Array
    segments: jax.Array
    location_iterations: jax.Array
    transitions: jax.Array


def _keep[T](value: T) -> T:
    return value


def _nominal_substep[S](system: GuardedSystem[S], sub_dt: jax.Array, budget: AdvanceBudget) -> Callable[[int, _Progress[S]], _Progress[S]]:
    """One nominal substep: bounded segment attempts from the accepted state, then the projection once.

    Without events the first trial, over exactly `sub_dt`, runs outside any conditional and the bounded
    retries only when it is rejected: wrapping the ordinary trial in control flow changes how XLA fuses
    and rounds it, and the ordinary path must reproduce the unguarded RK4 bit for bit.
    """

    def segments(segment: _Segment[S]) -> _Segment[S]:
        def attempt(_: int, current: _Segment[S]) -> _Segment[S]:
            pending = ~current.done & (current.status == AdvanceStatus.SUCCESS)
            return jax.lax.cond(pending, _attempt(system, budget), _keep, current)

        return jax.lax.fori_loop(0, budget.attempts, attempt, segment)

    def run(_: int, progress: _Progress[S]) -> _Progress[S]:
        active = progress.status == AdvanceStatus.SUCCESS
        start = _Segment(
            state=progress.state,
            context=progress.context,
            quadratures=progress.quadratures,
            remaining=sub_dt,
            h_try=sub_dt,
            accepted=jnp.zeros_like(sub_dt),
            done=jnp.zeros((), bool),
            status=progress.status,
            reason=progress.reason,
            event_index=progress.event_index,
            events=jnp.zeros((), jnp.int32),
            retries=progress.retries,
            segments=progress.segments,
            location_iterations=progress.location_iterations,
            transitions=progress.transitions,
        )
        if system.registry.events:
            end = jax.lax.cond(active, segments, _keep, start)
        else:
            first = rk4_trial(system, progress.state, progress.context, sub_dt)
            first_ok = ~first.check.hard & ~first.check.continuation
            tried = _commit_or_reject(replace(start, segments=start.segments + 1), first, sub_dt, first_ok)
            end = jax.lax.cond(active & ~first_ok, segments, _keep, tried)

        unfinished = (end.status == AdvanceStatus.SUCCESS) & ~end.done
        status = jnp.where(unfinished, AdvanceStatus.RETRY_EXHAUSTED, end.status)
        completed = progress.completed + end.accepted

        projected = system.project(end.state)
        projection_check = system.check(projected, end.context)
        projection_failed = projection_check.hard | projection_check.continuation
        quadratures = end.quadratures
        if system.observe_projection is not None:
            observed = system.observe_projection(end.state, projected)
            quadratures = {**quadratures, PROJECTION: jax.tree.map(jnp.add, quadratures[PROJECTION], observed)}

        finished = status == AdvanceStatus.SUCCESS
        accepted = finished & ~projection_failed
        advanced = _Progress(
            state=_tree_where(accepted, projected, end.state),
            context=end.context,
            quadratures=_tree_where(accepted, quadratures, end.quadratures),
            status=jnp.where(finished & projection_failed, AdvanceStatus.INVALID_STAGE, status).astype(jnp.int32),
            reason=jnp.where(finished & projection_failed, projection_check.reason, end.reason).astype(jnp.int32),
            event_index=end.event_index,
            completed=jnp.where(accepted, progress.completed + sub_dt, completed),
            retries=end.retries,
            segments=end.segments,
            location_iterations=end.location_iterations,
            transitions=end.transitions,
        )
        return _tree_where(active, advanced, progress)

    return run


def _attempt[S](system: GuardedSystem[S], budget: AdvanceBudget) -> Callable[[_Segment[S]], _Segment[S]]:
    """One trial from the accepted state: commit it, split it at the earliest event, or reject and halve it."""
    registry = system.registry

    def trial_from(segment: _Segment[S]) -> _Segment[S]:
        h = jnp.minimum(segment.h_try, segment.remaining)
        trial = rk4_trial(system, segment.state, segment.context, h)
        acceptable = ~trial.check.hard & ~trial.check.continuation
        if not registry.events:
            return _commit_or_reject(segment, trial, h, acceptable)

        surfaces = _surfaces(system, segment.state, segment.context, h)
        active = registry.active(segment.context) & ~trial.check.hard
        zero_time, crossing = _candidates(registry, active, surfaces, _surfaces(system, trial.state, segment.context, h))

        def ordinary(current: _Segment[S]) -> _Segment[S]:
            return _commit_or_reject(current, trial, h, acceptable)

        def at_once(current: _Segment[S]) -> _Segment[S]:
            return _apply_events(registry, budget, current, zero_time, jnp.argmax(zero_time))

        def split(current: _Segment[S]) -> _Segment[S]:
            return _locate_and_split(system, budget, current, h, surfaces, crossing)

        branch = jnp.where(jnp.any(zero_time), 1, jnp.where(jnp.any(crossing), 2, 0))
        return jax.lax.switch(branch, (ordinary, at_once, split), segment)

    def run(segment: _Segment[S]) -> _Segment[S]:
        segment = replace(segment, segments=segment.segments + 1)
        if not registry.events:
            return trial_from(segment)
        segment = _fire_past_events(system, budget, segment)
        return jax.lax.cond(segment.status == AdvanceStatus.SUCCESS, trial_from, _keep, segment)

    return run


def _commit_or_reject[S](segment: _Segment[S], trial: TrialResult[S], h: jax.Array, acceptable: jax.Array) -> _Segment[S]:
    remaining = segment.remaining - h
    return replace(
        segment,
        state=_tree_where(acceptable, trial.state, segment.state),
        quadratures=_commit(segment.quadratures, trial.increments, acceptable),
        remaining=jnp.where(acceptable, remaining, segment.remaining),
        h_try=jnp.where(acceptable, remaining, h * 0.5),
        accepted=jnp.where(acceptable, segment.accepted + h, segment.accepted),
        done=acceptable & (remaining <= 0),
        status=jnp.where(trial.origin_failed, AdvanceStatus.INVALID_STAGE, segment.status).astype(jnp.int32),
        reason=jnp.where(acceptable, segment.reason, trial.check.reason).astype(jnp.int32),
        retries=segment.retries + (~acceptable).astype(jnp.int32),
    )


def _commit(
    quadratures: Mapping[str, Mapping[str, jax.Array]], increments: Mapping[str, Mapping[str, jax.Array]], accepted: jax.Array
) -> dict[str, Mapping[str, jax.Array]]:
    """Add a trial's ledger increments when it is accepted; projection records accrue separately."""
    ledgers = {name: quadratures[name] for name in increments}
    return {**quadratures, **commit_if_accepted(ledgers, dict(increments), accepted)}


def _fire_past_events[S](system: GuardedSystem[S], budget: AdvanceBudget, segment: _Segment[S]) -> _Segment[S]:
    """Apply, at zero time, transitions whose surface already lies strictly past its crossing, as when inputs changed between calls."""
    registry = system.registry
    terminal = _terminal(registry.events)

    def past(context: IntegrationContext) -> jax.Array:
        return _past(registry, _surfaces(system, segment.state, context, segment.remaining))

    chain = run_zero_time_chain(registry, segment.context, lambda context: past(context) & ~terminal, budget.transitions)
    depleted = jnp.any(past(chain.context) & registry.active(chain.context) & terminal)
    status = jnp.select(
        [chain.incompatible, chain.exhausted, depleted],
        [AdvanceStatus.EVENT_LOCATION_FAILED, AdvanceStatus.EVENT_BUDGET_EXHAUSTED, AdvanceStatus.DEPLETED],
        segment.status,
    )
    return replace(segment, context=chain.context, status=status.astype(jnp.int32), transitions=segment.transitions + chain.transitions)


def _locate_and_split[S](
    system: GuardedSystem[S], budget: AdvanceBudget, segment: _Segment[S], h: jax.Array, surfaces: jax.Array, crossing: jax.Array
) -> _Segment[S]:
    """Bracket each crossing over complete trials from the accepted origin, then accept the segment ending at the earliest.

    The segment ends on the far side of the crossing when that endpoint is acceptable, and otherwise on
    the near side, so a crossed continuation boundary is never accepted.
    """
    registry = system.registry
    brackets = [_bracket(system, budget, segment, h, surfaces, crossing, index) for index in range(len(registry.events))]
    located = jnp.stack([(bracket.status == LocatorStatus.LOCATED) & crossing[index] for index, bracket in enumerate(brackets)])
    tie = min(event.spec.tolerance.time_h for event in registry.events)
    selection = select_earliest(located, jnp.stack([bracket.lower_h for bracket in brackets]), tie)
    upper_h = jnp.stack([bracket.upper_h for bracket in brackets])[selection.index]
    lower_h = jnp.stack([bracket.lower_h for bracket in brackets])[selection.index]

    upper = rk4_trial(system, segment.state, segment.context, upper_h)
    upper_ok = ~upper.check.hard & ~upper.check.continuation

    def far_side(_: None) -> TrialResult[S]:
        return upper

    def near_side(_: None) -> TrialResult[S]:
        return rk4_trial(system, segment.state, segment.context, lower_h)

    chosen = jax.lax.cond(upper_ok, far_side, near_side, None)
    chosen_ok = ~chosen.check.hard & ~chosen.check.continuation
    h_event = jnp.where(upper_ok, upper_h, lower_h)
    accepted = selection.any & chosen_ok
    remaining = segment.remaining - h_event

    split = replace(
        segment,
        state=_tree_where(accepted, chosen.state, segment.state),
        quadratures=_commit(segment.quadratures, chosen.increments, accepted),
        remaining=jnp.where(accepted, remaining, segment.remaining),
        h_try=jnp.where(accepted, remaining, segment.h_try),
        accepted=jnp.where(accepted, segment.accepted + h_event, segment.accepted),
        status=jnp.where(accepted, segment.status, AdvanceStatus.EVENT_LOCATION_FAILED).astype(jnp.int32),
        reason=jnp.where(chosen_ok, segment.reason, chosen.check.reason).astype(jnp.int32),
        location_iterations=segment.location_iterations + sum((bracket.iterations for bracket in brackets), jnp.zeros((), jnp.int32)),
    )
    applied = _apply_events(registry, budget, split, selection.simultaneous & accepted, selection.index)
    return replace(applied, done=accepted & (remaining <= 0) & (applied.status == AdvanceStatus.SUCCESS))


def _bracket[S](
    system: GuardedSystem[S], budget: AdvanceBudget, segment: _Segment[S], h: jax.Array, surfaces: jax.Array, crossing: jax.Array, index: int
) -> Bracket:
    """Locate event `index` when it is a crossing candidate; every bisection sample is a full trial from the accepted origin."""
    event = system.registry.events[index]

    def trial(duration: jax.Array) -> Trial:
        candidate = rk4_trial(system, segment.state, segment.context, duration)
        return Trial(_surfaces(system, candidate.state, segment.context, duration)[index], ~candidate.check.hard)

    def locate(_: None) -> Bracket:
        return locate_crossing(trial, surfaces[index], h, event.spec.direction, event.spec.tolerance, budget.location_iterations)

    def skip(_: None) -> Bracket:
        return _no_crossing(h)

    return jax.lax.cond(crossing[index], locate, skip, None)


def _no_crossing(h: jax.Array) -> Bracket:
    zero = jnp.zeros_like(h)
    return Bracket(
        status=jnp.asarray(LocatorStatus.NO_CROSSING, jnp.int32),
        lower_h=zero,
        upper_h=h,
        lower_surface=zero,
        upper_surface=zero,
        event_h=zero,
        residual=zero,
        zero_time=jnp.zeros((), bool),
        iterations=jnp.zeros((), jnp.int32),
    )


def _apply_events[S](registry: EventRegistry, budget: AdvanceBudget, segment: _Segment[S], firing: jax.Array, index: jax.Array) -> _Segment[S]:
    """Apply the firing events' transitions in registry order, or end the advance if one is terminal, two collide, or events run out."""
    terminal = _terminal(registry.events)
    context = segment.context
    for position in range(len(registry.events)):

        def apply(current: IntegrationContext, position: int = position) -> IntegrationContext:
            return registry.apply(jnp.asarray(position), current)

        context = jax.lax.cond(firing[position] & ~terminal[position], apply, _keep, context)

    fired = jnp.any(firing)
    events = segment.events + fired.astype(jnp.int32)
    status = jnp.select(
        [registry.conflicts(firing), jnp.any(firing & terminal), events > budget.events],
        [AdvanceStatus.EVENT_LOCATION_FAILED, AdvanceStatus.DEPLETED, AdvanceStatus.EVENT_BUDGET_EXHAUSTED],
        segment.status,
    )
    return replace(
        segment,
        context=context,
        status=jnp.where(segment.status == AdvanceStatus.SUCCESS, status, segment.status).astype(jnp.int32),
        event_index=jnp.where(fired, index, segment.event_index).astype(jnp.int32),
        events=events,
        transitions=segment.transitions + jnp.sum(firing & ~terminal).astype(jnp.int32),
    )


def _terminal(events: tuple[RegisteredEvent, ...]) -> jax.Array:
    return jnp.asarray([event.spec.continuation is Continuation.TERMINAL for event in events], dtype=bool)


def _candidates(registry: EventRegistry, eligible: jax.Array, start: jax.Array, end: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Events firing at zero time (on their surface, leaving in their direction) and events crossing inside the trial."""
    zero_time, crossing = [], []
    for index, event in enumerate(registry.events):
        g0, g1 = start[index], end[index]
        on_surface = jnp.abs(g0) <= event.spec.tolerance.surface
        match event.spec.direction:
            case CrossingDirection.DOWN:
                leaves, armed = g1 < 0.0, g0 > 0.0
            case CrossingDirection.UP:
                leaves, armed = g1 > 0.0, g0 < 0.0
            case CrossingDirection.BOTH:
                leaves, armed = jnp.zeros((), bool), jnp.ones((), bool)
        zero_time.append(eligible[index] & on_surface & leaves)
        crossing.append(eligible[index] & ~on_surface & armed & (g0 * g1 <= 0.0))
    return jnp.stack(zero_time), jnp.stack(crossing)


def _past(registry: EventRegistry, surfaces: jax.Array) -> jax.Array:
    flags = []
    for index, event in enumerate(registry.events):
        g, tolerance = surfaces[index], event.spec.tolerance.surface
        match event.spec.direction:
            case CrossingDirection.DOWN:
                flags.append(g < -tolerance)
            case CrossingDirection.UP:
                flags.append(g > tolerance)
            case CrossingDirection.BOTH:
                flags.append(jnp.zeros((), bool))
    return jnp.stack(flags)


def _surfaces[S](system: GuardedSystem[S], state: S, context: IntegrationContext, like: jax.Array) -> jax.Array:
    if system.surfaces is None:
        raise ValueError("a system with events must evaluate their surfaces")
    return system.surfaces(state, context).astype(like.dtype)


def _zero_quadratures[S](system: GuardedSystem[S], state: S, context: IntegrationContext, h: jax.Array) -> dict[str, Mapping[str, jax.Array]]:
    def increments() -> Mapping[str, Mapping[str, jax.Array]]:
        return rk4_trial(system, state, context, h).increments

    quadratures = dict(_zeros(jax.eval_shape(increments)))
    if system.observe_projection is not None:
        if PROJECTION in quadratures:
            raise ValueError(f"a ledger cannot be named {PROJECTION!r}")
        quadratures[PROJECTION] = _zeros(jax.eval_shape(system.observe_projection, state, state))
    return quadratures


def _zeros[T](shapes: T) -> T:
    def zeros(shape: jax.ShapeDtypeStruct) -> jax.Array:
        return jnp.zeros(shape.shape, shape.dtype)

    return jax.tree.map(zeros, shapes)


def _first_failure(checks: tuple[StateCheck, ...]) -> StateCheck:
    hard = jnp.stack([check.hard for check in checks])
    continuation = jnp.stack([check.continuation for check in checks])
    failing = hard | continuation
    reasons = jnp.stack([check.reason for check in checks])
    return StateCheck(jnp.any(hard), jnp.any(continuation), jnp.where(jnp.any(failing), reasons[jnp.argmax(failing)], NO_REASON).astype(jnp.int32))


def _nonfinite(tree: object) -> jax.Array:
    leaves = [jnp.any(~jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]
    return jnp.any(jnp.stack(leaves)) if leaves else jnp.zeros((), bool)


def _tree_where[T](condition: jax.Array, on_true: T, on_false: T) -> T:
    def select(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.where(condition, a, b)

    return jax.tree.map(select, on_true, on_false)


def _tree_scale[S](tree: S, scalar: jax.Array) -> S:
    def scale(x: jax.Array) -> jax.Array:
        return x * scalar

    return jax.tree.map(scale, tree)


def _tree_add[S](tree_a: S, tree_b: S) -> S:
    def add(a: jax.Array, b: jax.Array) -> jax.Array:
        return a + b

    return jax.tree.map(add, tree_a, tree_b)
