from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.typing import VariableDict

ApplyFn = Callable[[VariableDict, jax.Array], tuple[jax.Array, ...]]


def qrc_loss(
    params: VariableDict,
    apply_fn: ApplyFn,
    states: jax.Array,
    next_states: jax.Array,
    rewards: jax.Array,
    discounts: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """QRC saddle loss: primal minimizes projected error, dual maximizes error detection.

    Works with any ``apply_fn`` whose first two outputs are ``(v, h)``, whether the
    dual probe reads a stop-gradiented shared trunk or an independently trained one.
    """
    v, h = apply_fn(params, states)[:2]
    v_next = apply_fn(params, next_states)[0]
    target = jax.lax.stop_gradient(rewards + discounts * v_next)
    delta = target - v

    v_loss = 0.5 * delta**2 + discounts * jax.lax.stop_gradient(h) * v_next
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - h) ** 2
    return jnp.mean(v_loss + h_loss), (delta, h)
