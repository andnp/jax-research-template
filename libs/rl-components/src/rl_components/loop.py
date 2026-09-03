"""The one shared training loop: environment interaction and boundary accounting.

``run`` drives an :class:`~rl_components.agent_protocol.AgentProtocol` against an
:class:`~rl_components.env_protocol.EnvProtocol` for a fixed horizon under a single
``jax.lax.scan``. Episode-boundary accounting, the reset, and the bootstrap
coefficient live here and nowhere else.

Indexing contract
-----------------
``step_index`` at scan iteration ``i`` is ``i``. The :class:`Timestep` handed to the
agent at ``i`` closes the transition opened at ``i - 1``. Iteration ``0`` therefore
closes nothing: its ``reward`` and ``discount`` are zero and its ``episode_end`` is
``False``, and an agent guards replay insertion and learning on ``step_index > 0``. A
run of ``N`` iterations yields ``N - 1`` closed transitions, because the action taken
on the final iteration opens a transition that never closes.

Loop metrics use the OTHER indexing, and the two differ by one. A loop metric
recorded at scan index ``i`` describes the environment transition produced by the
action taken at index ``i`` -- that is, the transition the agent will see CLOSED at
``i + 1``. So a run of ``N`` iterations records ``N`` environment transitions in
metrics while the agent observes ``N - 1`` of them closed. Aligning a ``loop/`` metric
with an agent metric from the same scan index compares a transition with the one
before it; mis-aligning them silently is easy, so shift deliberately.

Metric semantics
----------------
``loop/reward``, ``loop/discount``, ``loop/episode_end``, ``loop/terminated`` and
``loop/truncated`` are dense per-step signals.

``loop/episode_return`` and ``loop/episode_length`` are SPARSE IMPULSE signals. They
carry the completed episode's undiscounted return and step count on boundary steps and
are zero on every other step. Averaging them over a window without masking on
``loop/episode_end`` divides an episode's total by the window length and yields a
number with no meaning. When the horizon ends mid-episode the partial return and
length are never emitted as metrics at all; they remain in the returned
:class:`LoopState`.

Reaching ``steps`` mid-episode is neither a termination nor a truncation. It is simply
where data stops: ``episode_end`` is false on that step and the loop does nothing
special. Only the environment's own flags end an episode.

``gamma`` is a parameter of this loop and of nothing else. Two sources of truth for the
discount is precisely how the bootstrap bug class this contract exists to fix recurs.

Environment contract gap
------------------------
:class:`~rl_components.env_protocol.EnvProtocol` does not currently state it, but the
pytrees returned by ``reset`` and ``step`` must have identical structure, leaf shapes
and leaf dtypes. The boundary selection merges the two, and mismatched states cannot
be merged; the failure is a loud trace-time error.

Jitting
-------
``run`` is deliberately NOT decorated with ``jit``. ``agent``, ``env``, ``steps``,
``gamma``, ``truncation_policy`` and ``env_params`` are static or closed over, and only
``key`` is traced, so callers wrap it::

    final_state, metrics = jax.jit(
        lambda k: run(agent, env, k, steps=100_000, gamma=0.99)
    )(key)

Seed parallelism composes on top of the same wrapper, ``jax.vmap`` over a batch of
keys, per ADR 004.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from rl_components.agent_protocol import AgentProtocol
from rl_components.env_protocol import EnvProtocol, TruncationPolicy
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep, bootstrap_terms

LOOP_METRIC_PREFIX = "loop/"
"""Metric-key namespace reserved for the loop; an agent using it is an error."""


@chex_struct(frozen=True)
class LoopState[EnvStateT, AgentStateT, ObservationT]:
    """The scan carry: everything one loop iteration needs from the previous one.

    Attributes:
        env_state: The environment state to step from, post-reset at a boundary.
        agent_state: The agent state to step from.
        timestep: The view of the environment handed to the agent on the next
            iteration. It closes the transition opened by the previous action.
        episode_length: Steps taken in the current episode so far, int32. Reset to
            zero on a boundary step, so it is a partial count whenever the horizon
            ends mid-episode.
        episode_return: Undiscounted return accumulated in the current episode so far,
            float32. Reset to zero on a boundary step.
        key: PRNG key for the remaining iterations.
    """

    env_state: EnvStateT
    agent_state: AgentStateT
    timestep: Timestep[ObservationT]
    episode_length: jax.Array
    episode_return: jax.Array
    key: chex.PRNGKey


def run[ObsT, EnvStateT, ActT, ParamsT, AgentStateT](
    agent: AgentProtocol[AgentStateT, ObsT, ActT],
    env: EnvProtocol[ObsT, EnvStateT, ActT, ParamsT],
    key: chex.PRNGKey,
    *,
    steps: int,
    gamma: float,
    truncation_policy: TruncationPolicy | None = None,
    env_params: ParamsT | None = None,
) -> tuple[LoopState[EnvStateT, AgentStateT, ObsT], dict[str, jax.Array]]:
    """Drive one agent through one environment for a fixed horizon.

    Args:
        agent: The agent to drive. A static Python object, closed over by the scan
            body rather than passed as a traced value.
        env: The environment to drive it through. Also static.
        key: PRNG key. The only traced argument.
        steps: Number of scan iterations. The agent closes ``steps - 1`` transitions.
        gamma: Discount factor, used as the bootstrap coefficient wherever the
            bootstrap survives. This is its only home.
        truncation_policy: Overrides the environment spec's own policy. ``None``
            defers to ``env.spec(env_params).truncation_policy``.
        env_params: Environment parameters, forwarded to every ``env`` call.

    Returns:
        The final :class:`LoopState` and the scan-stacked metrics, whose leaves each
        have leading axis ``steps``. Metric keys beginning ``loop/`` are the loop's;
        the rest are the agent's.

    Raises:
        ValueError: If the resolved truncation policy is neither ``"bootstrap"`` nor
            ``"terminate"``, or if an agent metric key begins with ``loop/``.
    """
    spec = env.spec(env_params)
    policy: TruncationPolicy = spec.truncation_policy if truncation_policy is None else truncation_policy
    if policy not in ("bootstrap", "terminate"):
        raise ValueError(f"truncation_policy must be 'bootstrap' or 'terminate', got {policy!r}")

    key, agent_key, initial_reset_key = jax.random.split(key, 3)
    agent_state = agent.init(agent_key, spec)
    reset = env.reset(initial_reset_key, env_params)
    initial = LoopState(
        env_state=reset.state,
        agent_state=agent_state,
        timestep=Timestep(
            reward=jnp.zeros((), jnp.float32),
            discount=jnp.zeros((), jnp.float32),
            bootstrap_observation=reset.observation,
            episode_end=jnp.zeros((), jnp.bool_),
            observation=reset.observation,
        ),
        episode_length=jnp.zeros((), jnp.int32),
        episode_return=jnp.zeros((), jnp.float32),
        key=key,
    )

    def _body(
        loop_state: LoopState[EnvStateT, AgentStateT, ObsT],
        step_index: jax.Array,
    ) -> tuple[LoopState[EnvStateT, AgentStateT, ObsT], dict[str, jax.Array]]:
        carry_key, step_key, reset_key = jax.random.split(loop_state.key, 3)

        agent_step = agent.step(loop_state.agent_state, loop_state.timestep, step_index)
        env_step = env.step(step_key, loop_state.env_state, agent_step.action, env_params)

        discount, episode_end = bootstrap_terms(
            env_step.terminated,
            env_step.truncated,
            jnp.zeros((), jnp.bool_),
            gamma=gamma,
            truncation_policy=policy,
        )

        def _on_boundary() -> tuple[ObsT, EnvStateT]:
            fresh = env.reset(reset_key, env_params)
            return fresh.observation, fresh.state

        def _on_continuation() -> tuple[ObsT, EnvStateT]:
            return env_step.observation, env_step.state

        observation, env_state = jax.lax.cond(episode_end, _on_boundary, _on_continuation)

        reward = jnp.asarray(env_step.reward, jnp.float32)
        next_timestep = Timestep(
            reward=reward,
            discount=discount,
            bootstrap_observation=env_step.observation,
            episode_end=episode_end,
            observation=observation,
        )

        completed_length = (loop_state.episode_length + 1).astype(jnp.int32)
        completed_return = loop_state.episode_return + reward

        next_state = LoopState(
            env_state=env_state,
            agent_state=agent_step.state,
            timestep=next_timestep,
            episode_length=jnp.where(episode_end, jnp.zeros((), jnp.int32), completed_length),
            episode_return=jnp.where(episode_end, jnp.zeros((), jnp.float32), completed_return),
            key=carry_key,
        )

        metrics = dict(agent_step.metrics)
        for metric_key in metrics:
            if metric_key.startswith(LOOP_METRIC_PREFIX):
                raise ValueError(
                    f"agent metric {metric_key!r} collides with the reserved {LOOP_METRIC_PREFIX!r} namespace"
                )
        terminated = jnp.asarray(env_step.terminated, jnp.bool_)
        metrics["loop/reward"] = reward
        metrics["loop/discount"] = discount
        metrics["loop/episode_end"] = episode_end
        metrics["loop/terminated"] = terminated
        metrics["loop/truncated"] = episode_end & ~terminated
        metrics["loop/episode_return"] = jnp.where(episode_end, completed_return, jnp.zeros((), jnp.float32))
        metrics["loop/episode_length"] = jnp.where(episode_end, completed_length, jnp.zeros((), jnp.int32))
        return next_state, metrics

    return jax.lax.scan(_body, initial, jnp.arange(steps, dtype=jnp.int32))
