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
    """Advance tank level by one time step.

    Level is clipped to [min_level, max_level] to represent physical overflow/dry limits.
    Flow units must be consistent with cross_section_area and dt
    (e.g., flow in L/min, area in L/m, dt in min → level in m).
    """
    net_flow = inlet_flow - outlet_flow
    new_level = jnp.clip(
        state.level + (net_flow / params.cross_section_area) * dt,
        params.min_level,
        params.max_level,
    )
    return TankState(level=new_level)
