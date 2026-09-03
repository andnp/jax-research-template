"""The agent port: two methods, and the obligations the signatures cannot express.

An agent is ``init`` plus ``step``. Everything else the twelve private training loops
currently do -- environment interaction, episode-boundary accounting, the ``lax.scan``
horizon -- belongs to the loop, not to the agent.

The properties below follow from the port. None of them is checkable by a type checker,
and every one of them breaks at trace time rather than at type-check time, so they are
recorded here.

**The agent never sees ``env``.** It receives an :class:`~rl_components.env_protocol.EnvSpec`
at ``init`` and a :class:`~rl_components.timestep.Timestep` at ``step``, and nothing else.
A unit test can therefore drive ``step`` with a hand-built timestep and assert an exact
target, with no environment in the picture.

**``(s, a)`` stays in agent state.** The loop supplies ``(r, d, s')``; the agent
remembers what it did. This is what makes replay insertion the agent's business rather
than the loop's.

**Learning happens inside ``step``**, under a ``lax.cond`` on a can-train predicate.
There is no separate ``learn`` entry point to keep in sync with the acting path.

**``step_index`` counts transitions CLOSED so far**, not loop iterations, so
``LEARNING_STARTS``, ``t % TRAIN_FREQUENCY`` and epsilon schedules key off collected
experience. Note the asymmetry at both ends: iteration ``0`` closes nothing but opens the
first transition, which closes at iteration ``1``, so nothing is discarded at the start;
the action taken on the final iteration opens a transition that never closes, and that is
the one transition a run of ``N`` iterations does not record.

**THE AGENT OBJECT IS STATIC, NOT A JAX VALUE.** A Python object holding methods cannot
be a dynamic argument to a jitted ``run``. The implementation and its configuration are
closed over; only ``AgentStateT``, a pytree, moves through the scan carry. The corollary
is worth stating: anything ``step`` needs must be reachable from ``AgentStateT`` or closed
over. A network is reached through a ``TrainState.apply_fn`` static field; a discrete
action count is carried as an int32 leaf.

**``init`` receives no observation, so pending-transition slots must be ZERO-PRIMED from
``EnvSpec``.** Because a transition closes on the iteration after the action, agent state
carries ``last_obs`` and ``last_action`` (plus ``log_prob`` and ``value`` for PPO).
``init`` must allocate these as zeros of ``spec.observation_shape`` /
``spec.observation_dtype`` and ``spec.action_shape`` / ``spec.action_dtype``. Leaving them
unshaped is a JIT tracer shape error on the first ``step``; the zero values themselves are
never used, because insertion is guarded on ``step_index > 0``.

**``metrics`` NEEDS A FIXED KEY SCHEMA.** Every scan iteration must return the same keys
with the same leaf shapes and dtypes. An agent that emits a metric only on the iterations
where it learns, or that returns ``{}`` before learning starts, produces a pytree mismatch
under ``scan``. The schema is fixed at ``init``, and unavailable metrics are emitted as
zero placeholders.

**Keys beginning ``loop/`` are RESERVED for the training loop**, which reports reward,
discount, boundary flags and episode statistics under that prefix.

Why not RL-Glue's ``start() -> a``, ``step(r, s) -> a``, ``end(r)`` verbatim: under
``lax.scan`` every iteration must return an identically-shaped pytree, so an ``end`` that
returns no action is inexpressible. The dm_env encoding -- one ``step``, with boundary
information carried in the data -- is the JAX-compatible form of the same semantics. This
is an encoding change, not a semantic one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chex
import jax

from rl_components.env_protocol import EnvSpec
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep


@chex_struct(frozen=True)
class AgentStep[AgentStateT, ActionT]:
    """Everything one agent step produces.

    Attributes:
        state: The agent's state after acting, and after any learning that happened on
            this step. It is the scan carry, so its pytree structure, leaf shapes and
            leaf dtypes must be identical on every iteration.
        action: The action to apply to the environment, selected from
            ``Timestep.observation``.
        metrics: Diagnostics for this step, under a key schema fixed at ``init``.
            Unavailable metrics are zero placeholders rather than absent keys. Keys
            beginning ``loop/`` are reserved for the training loop.
    """

    state: AgentStateT
    action: ActionT
    metrics: dict[str, jax.Array]


@runtime_checkable
class AgentProtocol[AgentStateT, ObservationT, ActionT](Protocol):
    """The agent port: build a state from a spec, then advance it one timestep at a time.

    The implementation object itself is static under ``jit`` -- see the module docstring.
    """

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> AgentStateT:
        """Build the initial agent state.

        No observation is available yet, so pending-transition slots are zero-primed
        from ``spec``, and the metric key schema is fixed here.

        Args:
            key: PRNG key for parameter initialization and any other sampling.
            spec: The environment's shape, dtype and action-space description. The only
                thing the agent ever learns about the environment.

        Returns:
            The agent state that enters the loop's scan carry.
        """
        ...

    def step(
        self,
        state: AgentStateT,
        timestep: Timestep[ObservationT],
        step_index: jax.Array,
    ) -> AgentStep[AgentStateT, ActionT]:
        """Close the pending transition, then act.

        The normative ordering inside the body is: complete the pending transition using
        ``timestep.bootstrap_observation``, then reset traces if ``timestep.episode_end``,
        then select an action from ``timestep.observation``. Resetting traces first
        silently drops the last update of every episode.

        Args:
            state: The agent state from the previous step.
            timestep: The environment's view of this iteration: the first four fields
                close the transition begun by the previous action, and ``observation``
                begins the next one.
            step_index: Number of transitions closed so far, as an int32 scalar. It is
                ``0`` on the only iteration that closes nothing, so insertion and
                learning are guarded on ``step_index > 0``.

        Returns:
            The next state, the chosen action, and this step's metrics.
        """
        ...
