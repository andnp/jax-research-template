import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.disturbances.types import apply_disturbance_type
from process_control.transport import Transport


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
        return jax.tree.map(lambda d, o: jnp.where(active, d, o), disturbed, t)

    return jax.lax.fori_loop(0, max_events, body_fn, transport)
