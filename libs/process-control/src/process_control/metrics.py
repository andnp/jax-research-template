"""JAX-native physical metrics for process-control evaluation trajectories."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TrackingMetrics(NamedTuple):
    """Tracking statistics computed independently for each trailing axis."""

    iae: jnp.ndarray
    ise: jnp.ndarray
    maximum_overshoot: jnp.ndarray
    settled: jnp.ndarray
    settling_index: jnp.ndarray


class MovementMetrics(NamedTuple):
    """Movement statistics for realized actions."""

    delta: jnp.ndarray
    l1_total: jnp.ndarray
    l1_mean: jnp.ndarray
    l2_max: jnp.ndarray


class ActuatorMetrics(NamedTuple):
    """Saturation and rate-limit statistics for an actuator trajectory."""

    saturated: jnp.ndarray
    saturation_fraction: jnp.ndarray
    maximum_saturation_run: jnp.ndarray
    rate_limited: jnp.ndarray
    rate_limit_fraction: jnp.ndarray


class EventMetrics(NamedTuple):
    """Summary statistics for a boolean event trajectory."""

    event_count: jnp.ndarray
    fraction: jnp.ndarray
    maximum_run: jnp.ndarray


class ConstraintMetrics(NamedTuple):
    """Violation statistics for an explicitly bounded signal."""

    violation: jnp.ndarray
    event_count: jnp.ndarray
    fraction: jnp.ndarray
    maximum_run: jnp.ndarray
    maximum_excess: jnp.ndarray


def tracking_metrics(
    error: jnp.ndarray,
    *,
    dt: float | jnp.ndarray,
    settling_tolerance: float | jnp.ndarray,
) -> TrackingMetrics:
    """Compute time-weighted tracking and settling statistics."""
    error = jnp.asarray(error)
    if error.ndim < 1:
        raise ValueError("error must include a time axis")
    absolute_error = jnp.abs(error)
    within_tolerance = absolute_error <= settling_tolerance
    settled_from_here = jnp.flip(
        jnp.cumprod(jnp.flip(within_tolerance, axis=0), axis=0), axis=0
    ).astype(bool)
    has_settled = jnp.any(settled_from_here, axis=0)
    settling_index = jnp.argmax(settled_from_here, axis=0)
    horizon = error.shape[0]
    settling_index = jnp.where(has_settled, settling_index, horizon)
    return TrackingMetrics(
        iae=jnp.sum(absolute_error, axis=0) * dt,
        ise=jnp.sum(error**2, axis=0) * dt,
        maximum_overshoot=jnp.max(jnp.maximum(error, 0.0), axis=0),
        settled=has_settled,
        settling_index=settling_index,
    )


def movement_metrics(
    actions: jnp.ndarray,
    *,
    initial_action: jnp.ndarray | None = None,
) -> MovementMetrics:
    """Compute action movement from realized time-major actions."""
    actions = jnp.asarray(actions)
    if actions.ndim < 1:
        raise ValueError("actions must include a time axis")
    if initial_action is None:
        delta = jnp.concatenate(
            (jnp.zeros_like(actions[:1]), jnp.diff(actions, axis=0)), axis=0
        )
    else:
        initial = jnp.asarray(initial_action)
        delta = jnp.concatenate(
            (actions[:1] - initial, jnp.diff(actions, axis=0)), axis=0
        )
    movement = jnp.abs(delta)
    per_step_l1 = (
        jnp.sum(movement, axis=-1) if actions.ndim >= 3 else movement
    )
    per_step_l2 = (
        jnp.linalg.norm(delta, axis=-1) if actions.ndim >= 3 else movement
    )
    return MovementMetrics(
        delta=delta,
        l1_total=jnp.sum(per_step_l1, axis=0),
        l1_mean=jnp.mean(per_step_l1, axis=0),
        l2_max=jnp.max(per_step_l2, axis=0),
    )


def actuator_metrics(
    requested: jnp.ndarray,
    realized: jnp.ndarray,
    *,
    output_min: float | jnp.ndarray,
    output_max: float | jnp.ndarray,
    tolerance: float = 1e-5,
) -> ActuatorMetrics:
    """Compute saturation and command-realization differences."""
    requested = jnp.asarray(requested)
    realized = jnp.asarray(realized)
    if requested.shape != realized.shape:
        raise ValueError("requested and realized actions must have matching shapes")
    if requested.ndim < 1:
        raise ValueError("actions must include a time axis")
    saturated = (realized <= output_min) | (realized >= output_max)
    rate_limited = jnp.abs(realized - requested) > tolerance
    saturated_event = jnp.any(saturated, axis=-1) if realized.ndim >= 3 else saturated
    rate_limited_event = (
        jnp.any(rate_limited, axis=-1) if realized.ndim >= 3 else rate_limited
    )
    saturation_events = event_metrics(saturated_event)
    return ActuatorMetrics(
        saturated=saturated_event,
        saturation_fraction=saturation_events.fraction,
        maximum_saturation_run=saturation_events.maximum_run,
        rate_limited=rate_limited_event,
        rate_limit_fraction=jnp.mean(rate_limited_event, axis=0),
    )


def event_metrics(event: jnp.ndarray) -> EventMetrics:
    """Summarize boolean events over the time axis."""
    event = jnp.asarray(event, dtype=bool)
    if event.ndim < 1:
        raise ValueError("event must include a time axis")
    current = jnp.zeros_like(event[0], dtype=jnp.int32)

    def update(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        value: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        current, maximum = carry
        current = jnp.where(value, current + 1, 0)
        return (current, jnp.maximum(maximum, current)), None

    (_, maximum), _ = jax.lax.scan(update, (current, current), event)
    return EventMetrics(
        event_count=jnp.sum(event, axis=0),
        fraction=jnp.mean(event, axis=0),
        maximum_run=maximum,
    )


def constraint_metrics(
    signal: jnp.ndarray,
    *,
    lower: float | jnp.ndarray | None = None,
    upper: float | jnp.ndarray | None = None,
) -> ConstraintMetrics:
    """Summarize violations of explicit lower and upper signal limits."""
    if lower is None and upper is None:
        raise ValueError("at least one constraint bound is required")
    signal = jnp.asarray(signal)
    violation = jnp.zeros_like(signal, dtype=bool)
    if lower is not None:
        violation = violation | (signal < lower)
    if upper is not None:
        violation = violation | (signal > upper)
    events = event_metrics(violation)
    excess = jnp.zeros_like(signal)
    if lower is not None:
        excess = jnp.maximum(excess, lower - signal)
    if upper is not None:
        excess = jnp.maximum(excess, signal - upper)
    return ConstraintMetrics(
        violation=violation,
        event_count=events.event_count,
        fraction=events.fraction,
        maximum_run=events.maximum_run,
        maximum_excess=jnp.max(excess, axis=0),
    )


def integrated_cost(cost: jnp.ndarray, *, dt: float | jnp.ndarray) -> jnp.ndarray:
    """Integrate an explicitly supplied per-step economic cost."""
    cost = jnp.asarray(cost)
    if cost.ndim < 1:
        raise ValueError("cost must include a time axis")
    return jnp.sum(cost, axis=0) * dt


def nonfinite_count(*values: jnp.ndarray) -> jnp.ndarray:
    """Count non-finite values over time and any signal dimensions."""
    if not values:
        raise ValueError("at least one value is required")
    masks = [~jnp.isfinite(jnp.asarray(value)) for value in values]
    return jnp.sum(jnp.concatenate([jnp.ravel(mask) for mask in masks]))
