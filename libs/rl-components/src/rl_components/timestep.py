"""The environment-to-agent data contract.

A :class:`Timestep` is what the loop hands an agent on every iteration. Its five
fields split into two halves with different tenses: the first four **complete** the
transition begun by the agent's previous action, and the last one **begins** the next
transition.

Two boundary quantities are named rather than inferred from a single ``done`` flag,
because they answer different questions and disagree exactly once:

- ``discount`` is the bootstrap coefficient. It is ``0`` on termination and ``gamma``
  on **both** continuation and truncation.
- ``episode_end`` is the trajectory break. It is true on **both** termination and
  truncation.

They differ exactly on truncation. The rule is: bootstrap **at** a truncation, never
**across** one. A truncated episode's final state is a perfectly ordinary state of the
MDP whose value is worth estimating, but the trajectory stops there, so traces and
n-step accumulators must be cut.

The two observations split for the same reason:

- ``bootstrap_observation`` is the TRUE state the transition reached, and is never a
  post-reset observation. It is the state whose value the bootstrap must use.
- ``observation`` is the state to act from, and IS the post-reset observation at a
  boundary.

They are equal on every step except a boundary step.

There is deliberately no validity field. Iteration ``0`` is the only iteration that
closes no transition, so validity is exactly ``step_index > 0``; carrying a second
representation of that would only invite the two to diverge.

The agent's ordering within ``step`` is **normative**: complete the pending transition
using ``bootstrap_observation``, THEN reset traces if ``episode_end``, THEN select an
action from ``observation``. An agent that resets its traces first silently drops the
last update of every episode. A single ``step`` method under ``lax.scan`` cannot
enforce statement order structurally, and nothing in the types objects, so each agent
must guard this ordering in its own tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rl_components.env_protocol import TruncationPolicy
from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class Timestep[ObservationT]:
    """One loop iteration's view of the environment, as the agent sees it.

    Attributes:
        reward: Reward earned by the agent's previous action; ``0.0`` at iteration 0,
            which closes no transition.
        discount: Bootstrap coefficient for the closed transition: ``0`` on
            termination, ``gamma`` on continuation and on truncation alike.
        bootstrap_observation: The true state the closed transition reached. Never a
            post-reset observation, so a bootstrap taken from it is a bootstrap at the
            boundary rather than across it.
        episode_end: Whether the closed transition broke the trajectory, which covers
            termination and truncation alike.
        observation: The state to act from now. Equal to ``bootstrap_observation``
            except on a boundary step, where it is the post-reset observation.
    """

    # --- completes the transition begun by the agent's previous action ---
    reward: jax.Array
    discount: jax.Array
    bootstrap_observation: ObservationT
    episode_end: jax.Array
    # --- begins the next action ---
    observation: ObservationT


def bootstrap_terms(
    terminated: jax.Array,
    truncated: jax.Array,
    cutoff_reached: jax.Array,
    *,
    gamma: float,
    truncation_policy: TruncationPolicy,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map raw environment boundary flags to a ``Timestep``'s two boundary quantities.

    Termination dominates truncation. Both flags can be true on one step: a pole falls
    on exactly the step the cutoff is reached, or an adapter reports both. Dominance is
    enforced by which quantity drives ``kills_bootstrap``: it is ``is_terminated``, not
    the truncation flag, so a genuinely terminal transition -- whose final state has no
    meaningful value, because nothing follows it -- loses its bootstrap under either
    policy. The ``& ~is_terminated`` mask does not carry that rule; it makes the
    returned ``is_truncated`` flag mutually exclusive with termination, so a caller
    reporting both boundary kinds cannot report the same step as terminated and
    truncated at once.

    ``truncation_policy`` is a static Python string, so the branch on it resolves at
    trace time and only one bootstrap rule is ever staged out.

    Args:
        terminated: Whether the environment reached a terminal state.
        truncated: Whether the environment cut the episode at a time limit of its own.
        cutoff_reached: Whether the loop's own episode cutoff fired on this step.
        gamma: Discount factor to use wherever the bootstrap survives.
        truncation_policy: The task's answer to whether a truncation keeps its
            bootstrap (``"bootstrap"``) or kills it like a termination
            (``"terminate"``).

    Returns:
        The ``(discount, episode_end, is_truncated)`` triple: ``discount`` is float32
        and is ``0`` wherever the bootstrap is killed and ``gamma`` elsewhere;
        ``episode_end`` is bool and is true on termination and truncation alike;
        ``is_truncated`` is bool and is true only on a truncation that is not also a
        termination.

    Raises:
        ValueError: If ``truncation_policy`` is neither ``"bootstrap"`` nor
            ``"terminate"``. An unrecognised value would otherwise take the
            ``"terminate"`` branch silently and kill every truncation's bootstrap.
    """
    if truncation_policy not in ("bootstrap", "terminate"):
        raise ValueError(f"truncation_policy must be 'bootstrap' or 'terminate', got {truncation_policy!r}")

    is_terminated = jnp.asarray(terminated, dtype=jnp.bool_)
    is_truncated = (
        jnp.asarray(truncated, jnp.bool_) | jnp.asarray(cutoff_reached, jnp.bool_)
    ) & ~is_terminated
    episode_end = is_terminated | is_truncated
    kills_bootstrap = is_terminated if truncation_policy == "bootstrap" else episode_end
    discount = jnp.where(kills_bootstrap, jnp.zeros((), jnp.float32), jnp.asarray(gamma, jnp.float32))
    return discount, episode_end, is_truncated
