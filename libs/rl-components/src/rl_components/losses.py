from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.typing import VariableDict

ApplyFn = Callable[[VariableDict, jax.Array], tuple[jax.Array, ...]]


@jax.custom_vjp
def clip_gradient(value: jax.Array, lower: float, upper: float) -> jax.Array:
    """Return ``value`` while clipping its reverse-mode gradient."""
    return value


def _clip_gradient_fwd(value: jax.Array, lower: float, upper: float):
    return value, (lower, upper)


def _clip_gradient_bwd(residual: tuple[float, float], cotangent: jax.Array):
    lower, upper = residual
    return (jnp.clip(cotangent, lower, upper), None, None)


clip_gradient.defvjp(_clip_gradient_fwd, _clip_gradient_bwd)


def qrc_terms(
    v: jax.Array,
    h: jax.Array,
    reward: jax.Array,
    discount: jax.Array,
    bootstrap: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    target = jax.lax.stop_gradient(reward + discount * bootstrap)
    delta = target - v
    value_loss = 0.5 * delta**2 + discount * jax.lax.stop_gradient(h) * bootstrap
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - h) ** 2
    return value_loss, h_loss, delta


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
    v_loss, h_loss, delta = qrc_terms(v, h, rewards, discounts, v_next)
    return jnp.mean(v_loss + h_loss), (delta, h)
