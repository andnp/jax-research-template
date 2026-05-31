"""Blower actuator model for aeration control.

Models a variable-speed blower (with VFD) that converts a requested
oxygen transfer rate (kla) into an achieved kla, subject to:
  - Asymmetric ramp rates (spin-up ≠ coast-down)
  - Startup delay (VFD initialisation when going from off to on)
  - Min/max output clamping

Set ramp rates to large values (e.g. 1e6) and startup_delay to 0.0
for ideal (passthrough) behaviour.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BlowerParams:
    max_kla: float = 10.0       # h⁻¹
    max_ramp_up: float = 5.0    # h⁻¹ per hour (blower spin-up)
    max_ramp_down: float = 8.0  # h⁻¹ per hour (coast-down, typically faster)
    startup_delay: float = 0.05 # hours (~3 min VFD initialisation)


@dataclass(frozen=True)
class BlowerState:
    current_kla: jax.Array
    startup_remaining: jax.Array

    @staticmethod
    def create(initial_kla: float = 0.0) -> "BlowerState":
        return BlowerState(
            current_kla=jnp.array(initial_kla),
            startup_remaining=jnp.array(0.0),
        )


jax.tree_util.register_dataclass(
    BlowerState,
    data_fields=["current_kla", "startup_remaining"],
    meta_fields=[],
)


def reset(initial_kla: float, rng_key: jax.Array) -> BlowerState:
    return BlowerState.create(initial_kla)


def step(
    state: BlowerState,
    requested_kla: jax.Array,
    params: BlowerParams,
    dt: jax.Array,
) -> tuple[BlowerState, jax.Array]:
    # Startup delay: when going from off → on, impose pure delay
    was_off = state.current_kla <= 0.0
    wants_on = requested_kla > 0.0
    timer_expired = state.startup_remaining <= 0.0

    new_startup = jnp.where(
        was_off & wants_on & timer_expired,
        jnp.array(params.startup_delay),
        jnp.maximum(state.startup_remaining - dt, 0.0),
    )

    # During startup, blower can't move yet
    in_startup = new_startup > 0.0
    effective_request = jnp.where(in_startup, 0.0, requested_kla)

    # Asymmetric ramp limiting
    delta = effective_request - state.current_kla
    max_up = params.max_ramp_up * dt
    max_down = params.max_ramp_down * dt
    clamped_delta = jnp.where(
        delta > 0,
        jnp.minimum(delta, max_up),
        jnp.maximum(delta, -max_down),
    )

    new_kla = jnp.clip(state.current_kla + clamped_delta, 0.0, params.max_kla)

    return BlowerState(current_kla=new_kla, startup_remaining=new_startup), new_kla
