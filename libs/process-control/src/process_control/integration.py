"""Numerical integration methods for JAX pytree-valued ODEs.

All integrators operate on arbitrary JAX pytrees (frozen dataclasses,
nested arrays, etc.) via jax.tree.map. External inputs (params, flows,
kla) are held constant over the step.
"""

from collections.abc import Callable
from typing import Concatenate

import jax


def rk4_step[**P, S](
    derivative_fn: Callable[Concatenate[S, P], S],
    state: S,
    dt: jax.Array,
    *args: P.args,
    **kwargs: P.kwargs,
) -> S:
    """Advance state by one RK4 step.

    Args:
        derivative_fn: f(state, *args) -> state_derivative (same pytree structure)
        state: current state (arbitrary pytree)
        dt: time step
        *args: additional arguments passed to derivative_fn unchanged
    """
    k1 = derivative_fn(state, *args, **kwargs)
    k2 = derivative_fn(_tree_add(state, _tree_scale(k1, dt * 0.5)), *args, **kwargs)
    k3 = derivative_fn(_tree_add(state, _tree_scale(k2, dt * 0.5)), *args, **kwargs)
    k4 = derivative_fn(_tree_add(state, _tree_scale(k3, dt)), *args, **kwargs)

    # dy = (k1 + 2*k2 + 2*k3 + k4) / 6
    dy = jax.tree.map(
        lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0,
        k1,
        k2,
        k3,
        k4,
    )
    return _tree_add(state, _tree_scale(dy, dt))


def _tree_scale[S](tree: S, scalar: jax.Array) -> S:
    return jax.tree.map(lambda x: x * scalar, tree)


def _tree_add[S](tree_a: S, tree_b: S) -> S:
    return jax.tree.map(lambda a, b: a + b, tree_a, tree_b)
