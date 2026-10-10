"""Chains of identical first-order lags, the BSM1 response-time model for sensors and actuators.

Alex et al. (2008), Sec. 7.3.1: https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

import math

import jax
import jax.numpy as jnp


def time_constant_for_t90(t90: float, order: int) -> float:
    """Stage time constant at which a chain of `order` lags reaches 90 % of a step after t90."""
    if order < 1 or t90 <= 0:
        raise ValueError(f"a lag chain needs a positive order and T90, got order {order} and T90 {t90}")

    def step_response(ratio: float) -> float:
        return 1.0 - math.exp(-ratio) * sum(ratio**k / math.factorial(k) for k in range(order))

    low, high = 0.0, 10.0 * order + 10.0
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if step_response(middle) < 0.9 else (low, middle)
    return t90 / low


def advance_lag_chain(stages: jax.Array, target: jax.Array, time_constant: float, dt: jax.Array) -> jax.Array:
    """Exact response over dt of lag stages, input side first, to a target held constant."""
    order = stages.shape[0]
    ratio = dt / time_constant
    gap = jnp.arange(order)[:, None] - jnp.arange(order)[None, :]
    lag = jnp.maximum(gap, 0)
    factorials = jnp.array([math.factorial(k) for k in range(order)], dtype=stages.dtype)
    transition = jnp.where(gap >= 0, jnp.exp(-ratio) * ratio**lag / factorials[lag], 0.0)
    return target + transition @ (stages - target)
