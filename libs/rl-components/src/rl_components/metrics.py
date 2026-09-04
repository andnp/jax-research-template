"""Transforms over the metric dict returned by :func:`rl_components.loop.run`.

``run`` stacks metrics under two reserved namespaces, ``loop/`` and ``env/``, plus
whatever unprefixed keys the agent emits. The two functions here are the
key-renaming and impulse-forward-filling steps that every consumer of those
metrics has otherwise hand-rolled.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from rl_components.loop import ENV_METRIC_PREFIX


def project_metrics(
    metrics: Mapping[str, jax.Array],
    *,
    rename: Mapping[str, str] | None = None,
) -> dict[str, jax.Array]:
    """Flatten ``env/`` and ``loop/reward`` keys to the bare names ``experiment.add_metric`` expects.

    ``env/foo`` becomes ``foo`` and ``loop/reward`` becomes ``reward``, matching the one
    loop metric every consumer surfaces unprefixed. Every other ``loop/`` key keeps its
    prefix rather than being dropped, so it stays reachable without risking a collision
    with an agent-owned name nobody asked to rename. Pass `rename` to project additional
    keys (e.g. more ``loop/`` keys) the same way.
    """
    targets: dict[str, str] = {"loop/reward": "reward"}
    for key in metrics:
        if key.startswith(ENV_METRIC_PREFIX):
            targets[key] = key[len(ENV_METRIC_PREFIX) :]
    if rename is not None:
        targets.update(rename)

    projected: dict[str, jax.Array] = {}
    for key, value in metrics.items():
        target = targets.get(key, key)
        if target in projected:
            raise ValueError(f"metric {target!r} claimed by more than one namespace")
        projected[target] = value
    return projected


def episode_return_curve(metrics: Mapping[str, jax.Array], *, axis: int = -1) -> jax.Array:
    """Forward-fill ``loop/episode_return``'s sparse impulse into a dense return curve.

    ``loop/episode_return`` carries a completed episode's return only on the step it
    ends and zero elsewhere, because a fixed key schema under ``lax.scan`` cannot emit
    a value on some iterations only. This holds the most recently completed episode's
    return across every following step, and is zero before the first episode ends.
    `axis` names the time axis and defaults to the last one, so a single run's
    ``(steps,)`` and a batched ``(seeds, steps)`` input both work unchanged; the
    result has the same shape as the input.
    """
    episode_return = jnp.moveaxis(metrics["loop/episode_return"], axis, 0)
    episode_end = jnp.moveaxis(metrics["loop/episode_end"], axis, 0)

    def hold(previous: jax.Array, step: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value, ended = step
        current = jnp.where(ended, value, previous)
        return current, current

    initial = jnp.zeros(episode_return.shape[1:], dtype=episode_return.dtype)
    _, curve = jax.lax.scan(hold, initial, (episode_return, episode_end))
    return jnp.moveaxis(curve, 0, axis)
