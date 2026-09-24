"""Shared replay-backed agent state and step orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import chex
import jax
import jax.numpy as jnp

from rl_components.agent_protocol import AgentStep
from rl_components.continuous_actions import uniform_exploration_action
from rl_components.env_protocol import EnvSpec
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep


class _CountedReplayState(Protocol):
    @property
    def count(self) -> jax.Array: ...


@chex_struct(frozen=True)
class ReplayAgentState[AlgorithmStateT, ReplayStateT: _CountedReplayState]:
    """Common carry fields for an agent that learns from replay."""

    algorithm_state: AlgorithmStateT
    replay_state: ReplayStateT
    last_obs: jax.Array
    last_action: jax.Array
    key: chex.PRNGKey


def init_replay_agent_state[AlgorithmStateT, ReplayStateT: _CountedReplayState](
    algorithm_state: AlgorithmStateT,
    replay_state: ReplayStateT,
    key: chex.PRNGKey,
    spec: EnvSpec,
) -> ReplayAgentState[AlgorithmStateT, ReplayStateT]:
    """Attach replay storage and zero-primed transition slots to learner state."""
    return ReplayAgentState(
        algorithm_state=algorithm_state,
        replay_state=replay_state,
        last_obs=jnp.zeros(spec.observation_shape, dtype=spec.observation_dtype),
        last_action=jnp.zeros(spec.action_shape, dtype=spec.action_dtype),
        key=key,
    )


def replay_agent_step[AlgorithmStateT, ReplayStateT: _CountedReplayState](
    state: ReplayAgentState[AlgorithmStateT, ReplayStateT],
    timestep: Timestep[jax.Array],
    step_index: jax.Array,
    *,
    carry_key: chex.PRNGKey,
    learn_keys: tuple[jax.Array, ...],
    action_keys: tuple[jax.Array, ...],
    learning_starts: int | jax.Array | None,
    train_frequency: int | jax.Array,
    warmup_steps: int | jax.Array | None,
    close_transition: Callable[[ReplayStateT, jax.Array, jax.Array, Timestep[jax.Array]], ReplayStateT],
    learn: Callable[
        [AlgorithmStateT, ReplayStateT, tuple[jax.Array, ...]],
        tuple[AlgorithmStateT, ReplayStateT, dict[str, jax.Array]],
    ],
    select_action: Callable[
        [AlgorithmStateT, jax.Array, jax.Array, tuple[jax.Array, ...]],
        tuple[jax.Array, dict[str, jax.Array]],
    ],
    minimum_replay_size: int = 0,
    after_learn: Callable[[AlgorithmStateT, jax.Array, jax.Array], AlgorithmStateT] | None = None,
) -> AgentStep[ReplayAgentState[AlgorithmStateT, ReplayStateT], jax.Array]:
    """Close replay, conditionally learn, apply periodic updates, then act.

    ``close_transition`` receives the replay state, pending observation and action, plus
    this timestep; it can store one transition or update a multi-step accumulator. The
    helper guards it at index zero. Training is gated by ``learning_starts`` when
    provided, replay occupancy, and ``train_frequency``.
    ``after_learn`` runs after that decision on every step, with the decision passed as
    ``did_learn`` so target synchronization can also run on its own schedule.

    When ``warmup_steps`` is provided, the helper selects a uniform continuous action
    before that index. In that case ``action_keys`` must contain its action key. Otherwise
    ``select_action`` handles the algorithm's exploration policy and reports any
    action-derived metrics on every step.
    """
    if warmup_steps is not None and not action_keys:
        raise ValueError("action_keys must contain a key when uniform warmup is enabled")

    def close_pending(_: None) -> ReplayStateT:
        return close_transition(
            state.replay_state,
            state.last_obs,
            state.last_action,
            timestep,
        )

    def skip_close(_: None) -> ReplayStateT:
        return state.replay_state

    replay_state = jax.lax.cond(
        step_index > 0,
        close_pending,
        skip_close,
        operand=None,
    )
    algorithm_state = state.algorithm_state

    can_train = (step_index % train_frequency == 0) & (replay_state.count >= minimum_replay_size)
    if learning_starts is not None:
        can_train = can_train & (step_index > learning_starts)

    def train(_: None) -> tuple[AlgorithmStateT, ReplayStateT, dict[str, jax.Array]]:
        return learn(algorithm_state, replay_state, learn_keys)

    metrics_shape = jax.eval_shape(learn, algorithm_state, replay_state, learn_keys)[2]

    def zero_metric(metric: jax.ShapeDtypeStruct) -> jax.Array:
        return jnp.zeros(metric.shape, dtype=metric.dtype)

    metrics_when_not_learning = jax.tree.map(
        zero_metric,
        metrics_shape,
    )

    def skip_train(_: None) -> tuple[AlgorithmStateT, ReplayStateT, dict[str, jax.Array]]:
        return algorithm_state, replay_state, metrics_when_not_learning

    algorithm_state, replay_state, metrics = jax.lax.cond(can_train, train, skip_train, operand=None)

    if after_learn is not None:
        algorithm_state = after_learn(algorithm_state, step_index, can_train)

    def choose_action() -> tuple[jax.Array, dict[str, jax.Array]]:
        return select_action(algorithm_state, timestep.observation, step_index, action_keys)

    if warmup_steps is None:
        action, action_metrics = choose_action()
    else:
        action_metrics_shape = jax.eval_shape(
            select_action,
            algorithm_state,
            timestep.observation,
            step_index,
            action_keys,
        )[1]
        zero_action_metrics = jax.tree.map(zero_metric, action_metrics_shape)

        def explore() -> tuple[jax.Array, dict[str, jax.Array]]:
            action = uniform_exploration_action(action_keys[0], state.last_action.shape, state.last_action.dtype)
            return action, zero_action_metrics

        action, action_metrics = jax.lax.cond(step_index < warmup_steps, explore, choose_action)

    metrics = {**metrics, **action_metrics}

    return AgentStep(
        state=ReplayAgentState(
            algorithm_state=algorithm_state,
            replay_state=replay_state,
            last_obs=timestep.observation,
            last_action=action,
            key=carry_key,
        ),
        action=action,
        metrics=metrics,
    )
