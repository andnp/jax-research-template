"""DQN Zoo's Atari DQN as an :class:`~rl_components.agent_protocol.AgentProtocol`.

The agent owns its Nature CNN, its replay buffer, its centered-RMSProp optimizer and its
exploration schedule. It owns no environment, no ``lax.scan`` and no discount:
:func:`rl_components.loop.run` supplies the horizon and ``gamma``.

``ADDITIONAL_DISCOUNT`` is deleted rather than renamed. ``Timestep.discount`` already
carries ``{0, gamma}``, so a second 0.99 factor beside ``gamma = 0.99`` would discount at
0.9801 -- the same two-sources-of-truth defect the port exists to remove.

Every frame-counted DQN Zoo period is resolved to env steps once, at construction, because
those conversions are Python-integer arithmetic that must not be restaged per iteration.
The schedules then key off ``step_index``, the count of transitions CLOSED, which is what
the old private loop's ``env_step`` counted: one insertion per iteration means
``buffer_state.count == step_index`` at the gate.

Three things ``step`` needs are spec-derived, and ``step`` never sees the spec -- it
receives only a :class:`~rl_components.timestep.Timestep` -- while the agent object itself
is static under ``jit`` and so cannot be mutated by ``init``. Each is therefore reachable
from the state instead:

- the network, through ``state.train_state.apply_fn``, which Flax marks
  ``pytree_node=False`` and which therefore survives the scan carry as a static field;
- the action count, as an int32 leaf, so a traced bound can bound random exploration;
- the replay buffer, rebuilt inside ``step`` from the shapes and dtypes of its own state.
  ``ReplayBuffer`` holds no data, only that geometry, so reconstructing it is free and
  needs nothing the state does not already carry.

The can-train predicate gates on ``buffer_state.count`` rather than on the DQN family's
``step_index > LEARNING_STARTS``. The two disagree about whether a wrapped buffer still
counts as warm, and reconciling them here would destroy the port's equivalence signal, so
this agent keeps the predicate it already had.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.heads import epsilon_greedy_action
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

from rl_agents.q_networks import NatureQNetwork, infer_nature_observation_layout


@chex_struct(frozen=True, kw_only=True)
class DQNAtariConfig:
    REPLAY_CAPACITY: int = 1_000_000
    MIN_REPLAY_CAPACITY_FRACTION: float = 0.05
    BATCH_SIZE: int = 32
    NUM_ACTION_REPEATS: int = 4
    TARGET_NETWORK_UPDATE_PERIOD_FRAMES: int = 40_000
    LEARN_PERIOD_FRAMES: int = 16
    LEARNING_RATE: float = 0.00025
    OPTIMIZER_EPSILON: float = 0.01 / 32**2
    RMSPROP_DECAY: float = 0.95
    RMSPROP_CENTERED: bool = True
    EXPLORATION_EPSILON_BEGIN: float = 1.0
    EXPLORATION_EPSILON_END: float = 0.1
    EXPLORATION_EPSILON_DECAY_FRAME_FRACTION: float = 0.02


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


def dqn_zoo_atari_frames_to_env_steps(frames: int, num_action_repeats: int) -> int:
    if frames < 0:
        raise ValueError("frames must be non-negative.")
    if num_action_repeats <= 0:
        raise ValueError("num_action_repeats must be positive.")
    if frames % num_action_repeats != 0:
        raise ValueError(
            "DQN Zoo frame-counted periods must divide evenly by num_action_repeats when converting to Atari env-step semantics."
        )
    return frames // num_action_repeats


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


def dqn_zoo_atari_exploration_epsilon(
    env_step: int,
    config: DQNAtariConfig,
    runtime_config: DQNAtariRuntimeConfig,
) -> float:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")

    min_replay_capacity = dqn_zoo_atari_min_replay_capacity(config)
    if env_step <= min_replay_capacity:
        return config.EXPLORATION_EPSILON_BEGIN

    decay_env_steps = dqn_zoo_atari_exploration_decay_env_steps(config, runtime_config)
    if decay_env_steps <= 0:
        raise ValueError("exploration decay must span at least one env step.")

    elapsed_decay_steps = min(env_step - min_replay_capacity, decay_env_steps)
    if elapsed_decay_steps >= decay_env_steps:
        return config.EXPLORATION_EPSILON_END

    progress = elapsed_decay_steps / decay_env_steps
    return config.EXPLORATION_EPSILON_BEGIN + (
        config.EXPLORATION_EPSILON_END - config.EXPLORATION_EPSILON_BEGIN
    ) * progress


def dqn_zoo_atari_should_learn(env_step: int, replay_size: int, config: DQNAtariConfig) -> bool:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")
    if replay_size < 0:
        raise ValueError("replay_size must be non-negative.")
    if replay_size < dqn_zoo_atari_min_replay_capacity(config):
        return False
    return env_step % dqn_zoo_atari_learn_period_env_steps(config) == 0


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
class DQNAtariAgentState:
    """Everything the Atari DQN agent carries between loop iterations.

    Attributes:
        train_state: Online parameters, optimizer state, and the static ``apply_fn``
            through which ``step`` reaches the Nature Q-network.
        target_params: Parameters of the target network the bootstrap is taken from.
        buffer_state: Replay contents. Its array shapes and dtypes are also the only
            record of the buffer's geometry.
        last_obs: Observation the pending transition started from. Zero-primed at
            ``init``; never read before the first insertion, which is guarded on
            ``step_index > 0``.
        last_action: Action that opened the pending transition, zero-primed likewise.
        num_actions: Size of the discrete action space, as an int32 leaf so that random
            exploration can sample against a traced bound.
        key: PRNG key for sampling minibatches and exploring.
        hypers: Swept hyperparameters, traced so a batch of arms shares one kernel.
    """

    train_state: TrainState
    target_params: VariableDict
    buffer_state: ReplayBufferState
    last_obs: jax.Array
    last_action: jax.Array
    num_actions: jax.Array
    key: chex.PRNGKey
    hypers: DQNAtariHypers


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
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = NatureQNetwork(
            action_dim=spec.num_actions,
            observation_layout=infer_nature_observation_layout(observation_shape),
        )

        params_key, carry_key = jax.random.split(key)
        params = network.init(params_key, jnp.zeros(observation_shape, dtype=observation_dtype))
        buffer = ReplayBuffer(
            capacity=self.config.REPLAY_CAPACITY,
            obs_shape=observation_shape,
            action_shape=tuple(spec.action_shape),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        return DQNAtariAgentState(
            train_state=TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=build_dqn_zoo_atari_rmsprop(self.config),
            ),
            target_params=params,
            buffer_state=buffer.init(),
            last_obs=jnp.zeros(observation_shape, dtype=observation_dtype),
            last_action=jnp.zeros(tuple(spec.action_shape), dtype=action_dtype),
            num_actions=jnp.asarray(spec.num_actions, dtype=jnp.int32),
            key=carry_key,
            hypers=dqn_atari_hypers(self.config),
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
        buffer = ReplayBuffer.from_state(state.buffer_state)
        carry_key, sample_key, action_key = jax.random.split(state.key, 3)

        buffer_state = jax.lax.cond(
            step_index > 0,
            lambda: buffer.add(
                state.buffer_state,
                state.last_obs[None, ...],
                state.last_action[None, ...],
                timestep.reward[None, ...],
                timestep.bootstrap_observation[None, ...],
                timestep.discount[None, ...],
            ),
            lambda: state.buffer_state,
        )

        hypers = state.hypers
        can_train = (buffer_state.count >= self.min_replay_capacity) & (step_index % self.learn_period_env_steps == 0)
        train_state, loss = jax.lax.cond(
            can_train,
            lambda: self._learn(state.train_state, state.target_params, buffer_state, buffer, sample_key, hypers),
            lambda: (state.train_state, jnp.zeros((), jnp.float32)),
        )

        target_params = jax.lax.cond(
            step_index % self.target_update_period_env_steps == 0,
            lambda: train_state.params,
            lambda: state.target_params,
        )

        epsilon = self._exploration_epsilon(step_index, hypers)
        q_values = jnp.asarray(train_state.apply_fn(train_state.params, timestep.observation))
        action = epsilon_greedy_action(q_values, epsilon, key=action_key).astype(state.last_action.dtype)

        return AgentStep(
            state=DQNAtariAgentState(
                train_state=train_state,
                target_params=target_params,
                buffer_state=buffer_state,
                last_obs=timestep.observation,
                last_action=action,
                num_actions=state.num_actions,
                key=carry_key,
                hypers=hypers,
            ),
            action=action,
            metrics={"loss": loss, "epsilon": epsilon},
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
        train_state: TrainState,
        target_params: VariableDict,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        hypers: DQNAtariHypers,
    ) -> tuple[TrainState, jax.Array]:
        """Take one gradient step on a replay minibatch.

        The rate, decay and epsilon are pushed into the optimizer state each step
        because the transformation itself is a static field and cannot hold a
        traced value.
        """
        opt_state = train_state.opt_state
        train_state = train_state.replace(
            opt_state=opt_state._replace(
                hyperparams={
                    **opt_state.hyperparams,
                    "learning_rate": hypers.LEARNING_RATE,
                    "decay": hypers.RMSPROP_DECAY,
                    "eps": hypers.OPTIMIZER_EPSILON,
                }
            ),
        )
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, key, self.config.BATCH_SIZE)

        def _loss_fn(params: VariableDict) -> jax.Array:
            q_values = jnp.asarray(train_state.apply_fn(params, obs))
            q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
            next_q_max = jnp.max(jnp.asarray(train_state.apply_fn(target_params, next_obs)), axis=-1)
            targets = rewards + discounts * next_q_max
            return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(targets)))

        loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
        return train_state.apply_gradients(grads=grads), loss.astype(jnp.float32)
