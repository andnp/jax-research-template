"""Iterated-GTD critic correction: a Bellman-iteration chain's Q/H loss.

Generalizes the single-head QRC saddle loss (``rl_agents.qrc.qrc_loss`` and its
siblings in ``sac_rc``/``greedy_ac``) to a chain of ``K`` heads, each bootstrapping
off another head's own (un-stopped) next-value and regressed by its own H-estimator
of that Bellman error. Covers both a target-free chain (every head online, head 0
self-bootstrapping) and an anchored chain (head 0 bootstraps off a frozen target with
no gradient to correct) through ``boot_idx``/``active`` rather than branching on mode.
"""

from __future__ import annotations

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp


def _first_bootstrap_reference(boot_idx: jax.Array) -> jax.Array:
    """Marks the first head, in index order, to bootstrap off each ``own_next_value`` source.

    A later head sharing that source (e.g. head 1 also bootstrapping off head 0's own value
    in a target-free chain, since head 0 self-bootstraps *and* feeds head 1's target) must
    not independently correct it too: two H-estimators each calibrated to correct one
    source's own residual do not add the way two chain-rule terms would, and doubling the
    correction there doubles the effective TDC coefficient (``c=2`` instead of ``c=1``),
    which can flip local stability. Masking every later reference to an already-corrected
    source keeps exactly one correction per dependency.
    """
    earlier_match = (boot_idx[:, None] == boot_idx[None, :]) & (jnp.arange(boot_idx.shape[0])[None, :] < jnp.arange(boot_idx.shape[0])[:, None])
    return ~jnp.any(earlier_match, axis=1)


def bootstrap_index(num_heads: int) -> jax.Array:
    """``m(k)``: which head's own next-value head ``k`` bootstraps off, for a target-free chain.

    ``m(0) = 0`` (head 0 self-bootstraps, there is no frozen predecessor);
    ``m(k) = k - 1`` for ``k >= 1`` (every later head bootstraps off its immediate
    predecessor's own value). An anchored chain uses ``jnp.arange(num_heads)`` (the
    identity) instead: index 0 of its ``own_next_value`` is the frozen target.
    """
    return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.arange(num_heads - 1, dtype=jnp.int32)])


class GTDIterationMetrics(NamedTuple):
    """Per-head Bellman-error breakdown for one transition."""

    q: jax.Array
    delta: jax.Array
    q_loss: jax.Array
    h_loss: jax.Array
    bellman_error_sum: jax.Array


def gtd_iteration_correction(
    own_next_value: jax.Array,
    q_online: jax.Array,
    h_online: jax.Array,
    active: jax.Array,
    boot_idx: jax.Array,
    reward: jax.Array,
    gamma: jax.Array,
) -> tuple[jax.Array, GTDIterationMetrics]:
    r"""Iterated-GTD's ``L_Q + L_H`` for one transition, one chain topology fits all.

    ``own_next_value[i]`` is head ``i``'s own raw next-state value -- the quantity that
    carries a real (non-stopped) gradient into head ``i``'s params via the next-state
    forward pass. ``boot_idx[k]`` selects which head's own value online head ``k``
    bootstraps off (:func:`bootstrap_index` for a target-free chain, or the identity
    for an anchored one, where index 0 of ``own_next_value`` is a frozen target).
    ``delta[k] = sg(y[k]) - q_online[k]`` is head ``k``'s actual (stop-gradiented)
    Bellman error.

    Each active head ``k``'s correction term ``gamma * own_next_value[boot_idx[k]] *
    sg(h_online[k])`` reintroduces the gradient ``sg(y[k])`` in the semi-gradient
    q-loss discards, scaled by ``h_online[k]``'s regressed estimate of that error --
    except when a later head shares an earlier head's bootstrap source (see
    :func:`_first_bootstrap_reference`), where only the first correction counts;
    every active head's H is still trained via ``h_loss`` regardless. ``active`` masks
    out heads with no live H-estimator at all -- e.g. an anchored chain's head 0, whose
    bootstrap source is a frozen target with no gradient to correct.
    """
    chex.assert_equal_shape([q_online, h_online, active, boot_idx])
    sg = jax.lax.stop_gradient

    bootstrap_value = own_next_value[boot_idx]
    y = reward + gamma * bootstrap_value
    delta = sg(y) - q_online
    q_loss = jnp.sum(0.5 * delta**2)

    corrects = active & _first_bootstrap_reference(boot_idx)
    correction = gamma * jnp.sum(jnp.where(corrects, bootstrap_value * sg(h_online), 0.0))
    h_loss = jnp.sum(jnp.where(active, 0.5 * (sg(delta) - h_online) ** 2, 0.0))

    metrics = GTDIterationMetrics(
        q=q_online, delta=delta, q_loss=q_loss, h_loss=h_loss, bellman_error_sum=jnp.sum(delta**2),
    )
    return q_loss + correction + h_loss, metrics
