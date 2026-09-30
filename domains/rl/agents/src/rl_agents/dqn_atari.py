"""DQN Zoo's Atari DQN as an :class:`~rl_components.agent_protocol.AgentProtocol`.

The agent owns its Nature CNN, its replay buffer, its centered-RMSProp optimizer and its
exploration schedule. It owns no environment, no ``lax.scan`` and no discount:
:func:`rl_components.loop.run` supplies the horizon and ``gamma``.

``ADDITIONAL_DISCOUNT`` is deleted rather than renamed. ``Timestep.discount`` already
carries ``{0, gamma}``, so a second 0.99 factor beside ``gamma = 0.99`` would discount at
0.9801 -- the same two-sources-of-truth defect the port exists to remove.

Every frame-counted DQN Zoo period is resolved to env steps once, at construction, because
those conversions are Python-integer arithmetic that must not be restaged per iteration.
The schedules key off ``step_index``, the count of transitions CLOSED, while the training
gate also checks replay occupancy. ``step`` receives only a :class:`~rl_components.timestep.Timestep`,
so the network and action count live in ``state.algorithm_state``. The frame buffer is
rebuilt from ``state.replay_state`` and the known frame count; it holds geometry, not data.

The can-train predicate uses replay occupancy rather than ``step_index > LEARNING_STARTS``.
The two disagree after the buffer wraps, so retaining the occupancy gate preserves the
original schedule.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import FrameReplayBuffer, FrameReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.heads import epsilon_greedy_action
from rl_components.losses import clip_gradient
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep
from rl_networks.q_networks import NatureQNetwork, infer_nature_observation_layout

from rl_agents.dqn_zoo_atari import DQNZooAtariConfig, dqn_zoo_atari_frames_to_env_steps


@chex_struct(frozen=True, kw_only=True)
class DQNAtariConfig(DQNZooAtariConfig):
    EXPLORATION_EPSILON_BEGIN: float = 1.0
    EXPLORATION_EPSILON_END: float = 0.1
    EXPLORATION_EPSILON_DECAY_FRAME_FRACTION: float = 0.02
    GRAD_ERROR_BOUND: float = 1 / 32


@chex_struct(frozen=True, kw_only=True)
class DQNAtariRuntimeConfig:
    TOTAL_TRAIN_ENV_STEPS: int = 50_000_000
    SEED: int = 42
    EVAL_EXPLORATION_EPSILON: float = 0.05


def build_dqn_zoo_atari_rmsprop(config: DQNAtariConfig) -> optax.GradientTransformation:
    return optax.inject_hyperparams(optax.rmsprop, static_args="centered")(
        learning_rate=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        decay=jnp.asarray(config.RMSPROP_DECAY, jnp.float32),
        eps=jnp.asarray(config.OPTIMIZER_EPSILON, jnp.float32),
        centered=config.RMSPROP_CENTERED,
    )


def dqn_atari_runtime_from_dqn_zoo(
    config: DQNAtariConfig,
    *,
    num_iterations: int = 200,
    num_train_frames_per_iteration: int = 1_000_000,
    seed: int = 42,
    eval_exploration_epsilon: float = 0.05,
) -> DQNAtariRuntimeConfig:
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative.")
    if num_train_frames_per_iteration < 0:
        raise ValueError("num_train_frames_per_iteration must be non-negative.")

    # DQN Zoo counts emulator frames, while this Atari runtime trains in env steps.
    # With action repeat, one env.step advances NUM_ACTION_REPEATS frames, so frame-based schedules must divide by that repeat count.
    total_train_frames = num_iterations * num_train_frames_per_iteration
    return DQNAtariRuntimeConfig(
        TOTAL_TRAIN_ENV_STEPS=dqn_zoo_atari_frames_to_env_steps(total_train_frames, config.NUM_ACTION_REPEATS),
        SEED=seed,
        EVAL_EXPLORATION_EPSILON=eval_exploration_epsilon,
    )


def dqn_zoo_atari_min_replay_capacity(config: DQNAtariConfig) -> int:
    return int(config.REPLAY_CAPACITY * config.MIN_REPLAY_CAPACITY_FRACTION)


def dqn_zoo_atari_learn_period_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.LEARN_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_target_update_period_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_total_train_frames(config: DQNAtariConfig, runtime_config: DQNAtariRuntimeConfig) -> int:
    total_train_env_steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
    return total_train_env_steps * config.NUM_ACTION_REPEATS


def dqn_zoo_atari_total_train_env_steps(runtime_config: DQNAtariRuntimeConfig) -> int:
    if runtime_config.TOTAL_TRAIN_ENV_STEPS < 0:
        raise ValueError("TOTAL_TRAIN_ENV_STEPS must be non-negative.")
    return runtime_config.TOTAL_TRAIN_ENV_STEPS


def dqn_zoo_atari_exploration_decay_env_steps(config: DQNAtariConfig, runtime_config: DQNAtariRuntimeConfig) -> int:
    decay_frames = int(
        dqn_zoo_atari_total_train_frames(config, runtime_config) * config.EXPLORATION_EPSILON_DECAY_FRAME_FRACTION
    )
    return dqn_zoo_atari_frames_to_env_steps(decay_frames, config.NUM_ACTION_REPEATS)


def dqn_atari_td_loss(q_action: jax.Array, targets: jax.Array, grad_error_bound: float) -> jax.Array:
    """Compute DQN's squared TD loss with a bounded TD-error gradient."""
    td_error = q_action - jax.lax.stop_gradient(targets)
    return jnp.mean(0.5 * clip_gradient(td_error, -grad_error_bound, grad_error_bound) ** 2)


@chex_struct(frozen=True)
class DQNAtariHypers:
    """The hyperparameters ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. Everything here is
    read only in arithmetic, never to size an array or take a Python branch --
    those stay on the config. ``LEARNING_RATE``, ``OPTIMIZER_EPSILON`` and
    ``RMSPROP_DECAY`` are also pushed into the optimizer state each learn step,
    because the optimizer transformation itself is a static field and cannot
    hold a traced value.
    """

    LEARNING_RATE: jax.Array
    OPTIMIZER_EPSILON: jax.Array
    RMSPROP_DECAY: jax.Array
    EXPLORATION_EPSILON_BEGIN: jax.Array
    EXPLORATION_EPSILON_END: jax.Array


def dqn_atari_hypers(config: DQNAtariConfig) -> DQNAtariHypers:
    return DQNAtariHypers(
        LEARNING_RATE=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        OPTIMIZER_EPSILON=jnp.asarray(config.OPTIMIZER_EPSILON, jnp.float32),
        RMSPROP_DECAY=jnp.asarray(config.RMSPROP_DECAY, jnp.float32),
        EXPLORATION_EPSILON_BEGIN=jnp.asarray(config.EXPLORATION_EPSILON_BEGIN, jnp.float32),
        EXPLORATION_EPSILON_END=jnp.asarray(config.EXPLORATION_EPSILON_END, jnp.float32),
    )


@chex_struct(frozen=True)
class DQNAtariAlgorithmState:
    train_state: TrainState
    target_params: VariableDict
    num_actions: jax.Array
    hypers: DQNAtariHypers


type DQNAtariAgentState = ReplayAgentState[DQNAtariAlgorithmState, FrameReplayBufferState]


class DQNAtariAgent:
    """DQN Zoo's Atari DQN: a Nature CNN, centered RMSProp, and a hard target copy."""

    def __init__(self, config: DQNAtariConfig, runtime_config: DQNAtariRuntimeConfig) -> None:
        """Bind the configurations and resolve every frame-counted schedule to env steps.

        Args:
            config: Learner hyperparameters. Read at trace time only, so every field may
                be a Python scalar.
            runtime_config: The declared run budget, which the exploration decay span is
                a fraction of.
        """
        self.config = config
        self.runtime_config = runtime_config
        self.min_replay_capacity = dqn_zoo_atari_min_replay_capacity(config)
        self.learn_period_env_steps = dqn_zoo_atari_learn_period_env_steps(config)
        self.target_update_period_env_steps = dqn_zoo_atari_target_update_period_env_steps(config)
        self.exploration_decay_env_steps = dqn_zoo_atari_exploration_decay_env_steps(config, runtime_config)

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> DQNAtariAgentState:
        """Build parameters, optimizer, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description. Must be discrete, and must carry image
                observations the Nature CNN can consume.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space.
        """
        if spec.num_actions is None:
            raise ValueError(f"DQN Atari requires a discrete action space, got spec {spec.id!r} with none")

        observation_shape = tuple(spec.observation_shape)
        self.observation_shape = observation_shape
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = NatureQNetwork(
            action_dim=spec.num_actions,
            observation_layout=infer_nature_observation_layout(observation_shape),
        )

        params_key, carry_key = jax.random.split(key)
        params = network.init(params_key, jnp.zeros(observation_shape, dtype=observation_dtype))
        buffer = FrameReplayBuffer(
            capacity=self.config.REPLAY_CAPACITY,
            obs_shape=observation_shape,
            action_shape=tuple(spec.action_shape),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        return init_replay_agent_state(
            DQNAtariAlgorithmState(
                train_state=TrainState.create(
                    apply_fn=network.apply,
                    params=params,
                    tx=build_dqn_zoo_atari_rmsprop(self.config),
                ),
                target_params=params,
                num_actions=jnp.asarray(spec.num_actions, dtype=jnp.int32),
                hypers=dqn_atari_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: DQNAtariAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[DQNAtariAgentState, jax.Array]:
        """Close the pending transition, learn from replay, sync the target, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so
                insertion is guarded above it, and every schedule keys off it.

        Returns:
            The next state, the action to apply, and a fixed two-key metric schema.
            ``loss`` is a zero placeholder on the iterations that do not learn, so the
            pytree returned to ``lax.scan`` is identical on every iteration.
        """
        buffer = FrameReplayBuffer.from_state(state.replay_state, self.observation_shape[0])
        carry_key, sample_key, action_key = jax.random.split(state.key, 3)

        def close_transition(
            replay_state: FrameReplayBufferState,
            last_obs: jax.Array,
            last_action: jax.Array,
            current_timestep: Timestep[jax.Array],
        ) -> FrameReplayBufferState:
            return buffer.add(
                replay_state,
                last_obs[None, ...],
                last_action[None, ...],
                current_timestep.reward[None, ...],
                current_timestep.bootstrap_observation[None, ...],
                current_timestep.discount[None, ...],
                current_timestep.episode_end[None, ...],
            )

        def after_learn(
            learner_state: DQNAtariAlgorithmState,
            current_step: jax.Array,
            _did_learn: jax.Array,
        ) -> DQNAtariAlgorithmState:
            return replace(
                learner_state,
                target_params=jax.lax.cond(
                    current_step % self.target_update_period_env_steps == 0,
                    lambda: learner_state.train_state.params,
                    lambda: learner_state.target_params,
                ),
            )

        def select_action(
            learner_state: DQNAtariAlgorithmState,
            observation: jax.Array,
            current_step: jax.Array,
            keys: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            epsilon = self._exploration_epsilon(current_step, learner_state.hypers)
            q_values = jnp.asarray(learner_state.train_state.apply_fn(learner_state.train_state.params, observation))
            action = epsilon_greedy_action(q_values, epsilon, key=keys[0]).astype(state.last_action.dtype)
            return action, {"epsilon": epsilon}

        return replay_agent_step(
            state,
            timestep,
            step_index,
            carry_key=carry_key,
            learn_keys=(sample_key,),
            action_keys=(action_key,),
            learning_starts=None,
            train_frequency=self.learn_period_env_steps,
            warmup_steps=None,
            close_transition=close_transition,
            learn=partial(self._learn, buffer=buffer),
            select_action=select_action,
            minimum_replay_size=self.min_replay_capacity,
            after_learn=after_learn,
        )

    def _exploration_epsilon(self, step_index: jax.Array, hypers: DQNAtariHypers) -> jax.Array:
        """Hold epsilon at its start value through the replay warmup, then decay linearly."""
        warmup_complete = step_index > self.min_replay_capacity
        elapsed_decay_steps = jnp.minimum(
            jnp.maximum(step_index - self.min_replay_capacity, 0),
            self.exploration_decay_env_steps,
        )
        progress = elapsed_decay_steps / self.exploration_decay_env_steps
        decayed = hypers.EXPLORATION_EPSILON_BEGIN + (
            hypers.EXPLORATION_EPSILON_END - hypers.EXPLORATION_EPSILON_BEGIN
        ) * progress
        return jnp.where(warmup_complete, decayed, hypers.EXPLORATION_EPSILON_BEGIN).astype(jnp.float32)

    def _learn(
        self,
        algorithm_state: DQNAtariAlgorithmState,
        replay_state: FrameReplayBufferState,
        keys: tuple[jax.Array, ...],
        *,
        buffer: FrameReplayBuffer,
    ) -> tuple[DQNAtariAlgorithmState, FrameReplayBufferState, dict[str, jax.Array]]:
        """Take one gradient step on a replay minibatch.

        The rate, decay and epsilon are pushed into the optimizer state each step
        because the transformation itself is a static field and cannot hold a
        traced value.
        """
        train_state = with_hyperparams(
            algorithm_state.train_state,
            learning_rate=algorithm_state.hypers.LEARNING_RATE,
            decay=algorithm_state.hypers.RMSPROP_DECAY,
            eps=algorithm_state.hypers.OPTIMIZER_EPSILON,
        )
        obs, actions, rewards, next_obs, discounts = buffer.sample(replay_state, keys[0], self.config.BATCH_SIZE)

        def _loss_fn(params: VariableDict) -> jax.Array:
            q_values = jnp.asarray(train_state.apply_fn(params, obs))
            q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
            next_q_max = jnp.max(jnp.asarray(train_state.apply_fn(algorithm_state.target_params, next_obs)), axis=-1)
            targets = rewards + discounts * next_q_max
            return dqn_atari_td_loss(q_action, targets, self.config.GRAD_ERROR_BOUND)

        loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
        algorithm_state = replace(algorithm_state, train_state=train_state.apply_gradients(grads=grads))
        return algorithm_state, replay_state, {"loss": loss.astype(jnp.float32)}
