"""Fixed-point steady-state analysis for constant-input process dynamics."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class EquilibriumResult[StateT]:
    """The state reached after relaxing a step function toward steady state."""

    state: StateT
    steps_taken: int
    converged: bool


def relax_to_equilibrium[StateT](
    step: Callable[[StateT], StateT],
    initial_state: StateT,
    max_steps: int,
    tolerance: float = 1e-6,
) -> EquilibriumResult[StateT]:
    """Advance `step` from `initial_state` until consecutive states stop changing.

    `step` must already close over any fixed inputs, parameters, and timestep, so
    a benchmark's constant-input steady-state response (equilibrium point, and
    how many steps it takes to reach it) can be characterized independent of its
    specific step signature. Useful for checking whether a calibrated or default
    parameter vector drives a state variable toward a physically sensible
    equilibrium, or toward one so extreme it saturates a derived quantity (e.g. a
    bounded sigmoid observation) long before the trace being fit ends.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    state = initial_state
    for steps_taken in range(1, max_steps + 1):
        next_state = step(state)
        delta = jnp.sqrt(
            sum(
                jnp.sum(jnp.square(next_leaf - leaf))
                for leaf, next_leaf in zip(
                    jax.tree_util.tree_leaves(state),
                    jax.tree_util.tree_leaves(next_state),
                    strict=True,
                )
            )
        )
        state = next_state
        if bool(delta < tolerance):
            return EquilibriumResult(state=state, steps_taken=steps_taken, converged=True)
    return EquilibriumResult(state=state, steps_taken=max_steps, converged=False)
