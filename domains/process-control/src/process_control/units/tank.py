from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class TankParams:
    max_level: float
    min_level: float
    cross_section_area: float  # volume per unit level (e.g., m2 or L/m)


@jax_dataclass
class TankState:
    level: jax.Array

    @staticmethod
    def create(initial_level: float) -> "TankState":
        return TankState(level=jnp.array(initial_level))


@jax_dataclass
class TankStepResult:
    """Tank state and the flows imposed by its inventory limits."""

    state: TankState
    realized_outlet_flow: jax.Array
    overflow_flow: jax.Array
    unmet_outlet_flow: jax.Array
    constraint_status: jax.Array


def reset(params: TankParams) -> TankState:
    initial = (params.max_level + params.min_level) / 2.0
    return TankState.create(initial)


def step(
    state: TankState,
    inlet_flow: jax.Array,
    outlet_flow: jax.Array,
    params: TankParams,
    dt: jax.Array,
) -> TankState:
    """Advance tank level, preserving the original state-only API."""
    return step_with_result(state, inlet_flow, outlet_flow, params, dt).state


def step_with_result(
    state: TankState,
    inlet_flow: jax.Array,
    requested_outlet_flow: jax.Array,
    params: TankParams,
    dt: jax.Array,
) -> TankStepResult:
    """Advance the tank and report flows constrained by available storage.

    Flow units must be consistent with cross_section_area and dt
    (e.g., flow in L/min, area in L/m, dt in min → level in m).

    ``constraint_status`` is -1 when available inventory limits the outlet,
    0 when unconstrained, and 1 when the tank overflows.
    """
    stored_volume = (state.level - params.min_level) * params.cross_section_area
    available_volume = jnp.maximum(stored_volume + inlet_flow * dt, 0.0)
    requested_outlet_volume = jnp.maximum(requested_outlet_flow, 0.0) * dt
    realized_outlet_volume = jnp.minimum(requested_outlet_volume, available_volume)
    unmet_outlet_volume = requested_outlet_volume - realized_outlet_volume

    volume_after_outlet = available_volume - realized_outlet_volume
    capacity = (params.max_level - params.min_level) * params.cross_section_area
    overflow_volume = jnp.maximum(volume_after_outlet - capacity, 0.0)
    final_volume = volume_after_outlet - overflow_volume

    new_state = TankState(
        level=params.min_level + final_volume / params.cross_section_area,
    )
    is_outlet_limited = unmet_outlet_volume > 0.0
    is_overflowing = overflow_volume > 0.0
    constraint_status = jnp.where(is_outlet_limited, -1, jnp.where(is_overflowing, 1, 0))

    return TankStepResult(
        state=new_state,
        realized_outlet_flow=realized_outlet_volume / dt,
        overflow_flow=overflow_volume / dt,
        unmet_outlet_flow=unmet_outlet_volume / dt,
        constraint_status=constraint_status,
    )
