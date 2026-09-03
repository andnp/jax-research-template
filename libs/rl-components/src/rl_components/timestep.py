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
enforce statement order structurally, and nothing in the types objects, so this
ordering is guarded by test rather than by construction.
"""

from __future__ import annotations

import jax

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
