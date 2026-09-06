import math
from numbers import Integral, Real

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.disturbances.types import (
    DISTURBANCE_DEMAND_SLUG,
    DISTURBANCE_NONE,
    DISTURBANCE_RAIN_STORM,
    apply_disturbance_type,
)
from process_control.transport import Transport

type DisturbanceEvent = tuple[int, int, float, int]


def validate_disturbance_events(
    max_events: int,
    events: tuple[DisturbanceEvent, ...],
) -> int:
    """Validate static disturbance data before it enters JAX state."""
    if isinstance(max_events, bool) or not isinstance(max_events, Integral):
        raise ValueError("max_disturbance_events must be an integer")
    max_events = int(max_events)
    if max_events <= 0:
        raise ValueError("max_disturbance_events must be positive")
    if not isinstance(events, tuple):
        raise ValueError("disturbance_events must be a tuple of event tuples")
    if len(events) > max_events:
        raise ValueError("disturbance_events exceeds max_disturbance_events")

    supported_types = {
        DISTURBANCE_NONE,
        DISTURBANCE_DEMAND_SLUG,
        DISTURBANCE_RAIN_STORM,
    }
    for index, event in enumerate(events):
        if not isinstance(event, tuple) or len(event) != 4:
            raise ValueError(f"disturbance event {index} must be (start_step, end_step, magnitude, type_id)")
        start_step, end_step, magnitude, type_id = event
        if isinstance(start_step, bool) or not isinstance(start_step, Integral):
            raise ValueError(f"disturbance event {index} start_step must be an integer")
        if isinstance(end_step, bool) or not isinstance(end_step, Integral):
            raise ValueError(f"disturbance event {index} end_step must be an integer")
        if start_step < 0:
            raise ValueError(f"disturbance event {index} start_step must be non-negative")
        if end_step <= start_step:
            raise ValueError(f"disturbance event {index} end_step must be after start_step")
        if isinstance(magnitude, bool) or not isinstance(magnitude, Real) or not math.isfinite(float(magnitude)):
            raise ValueError(f"disturbance event {index} magnitude must be a finite real number")
        if isinstance(type_id, bool) or not isinstance(type_id, Integral) or int(type_id) not in supported_types:
            raise ValueError(f"disturbance event {index} type_id is unsupported")
    return max_events


@jax_dataclass
class DisturbanceSchedule:
    start_steps: jax.Array
    end_steps: jax.Array
    magnitudes: jax.Array
    type_ids: jax.Array
    count: jax.Array


def create_empty(max_events: int = 16) -> DisturbanceSchedule:
    return DisturbanceSchedule(
        start_steps=jnp.zeros(max_events, dtype=jnp.int32),
        end_steps=jnp.zeros(max_events, dtype=jnp.int32),
        magnitudes=jnp.zeros(max_events, dtype=jnp.float32),
        type_ids=jnp.zeros(max_events, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def add_event(
    schedule: DisturbanceSchedule,
    start_step: int,
    end_step: int,
    magnitude: float,
    type_id: int,
) -> DisturbanceSchedule:
    max_events = schedule.start_steps.shape[0]
    idx = schedule.count % max_events
    return DisturbanceSchedule(
        start_steps=schedule.start_steps.at[idx].set(start_step),
        end_steps=schedule.end_steps.at[idx].set(end_step),
        magnitudes=schedule.magnitudes.at[idx].set(magnitude),
        type_ids=schedule.type_ids.at[idx].set(type_id),
        count=schedule.count + 1,
    )


def apply_active(schedule: DisturbanceSchedule, transport: Transport, current_step: jax.Array) -> Transport:
    max_events = schedule.start_steps.shape[0]

    def body_fn(i: int, t: Transport) -> Transport:
        active = (i < schedule.count) & (schedule.start_steps[i] <= current_step) & (current_step < schedule.end_steps[i])
        disturbed = apply_disturbance_type(t, schedule.type_ids[i], schedule.magnitudes[i])

        def select_disturbed(d: jax.Array, o: jax.Array) -> jax.Array:
            return jnp.where(active, d, o)

        return jax.tree.map(select_disturbed, disturbed, t)

    return jax.lax.fori_loop(0, max_events, body_fn, transport)
