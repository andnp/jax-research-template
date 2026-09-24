"""The scaffolding every replay-based Q agent shares, with the loss left open.

``dqn`` and ``double_q`` were the same 86-line ``step`` and the same 45-line ``init``,
differing in three lines of state-class name plus one expression: the target the replay
minibatch regresses towards. So the port that collapsed ``double_dqn`` into ``dueling_dqn``
left one level of the same duplication behind, and every property in the list below --
insertion under ``step_index > 0``, the ``can_train`` predicate, the target-network soft
update, the epsilon schedule, the metric schema -- existed in two live copies that had to
be corrected in lockstep.

They live here once, parameterised by two callables, so what remains in each agent module
is a network and a loss:

- :class:`~rl_agents.dqn.DQNAgent` binds :func:`~rl_agents.dqn.dqn_loss`;
- :class:`~rl_agents.double_q.DoubleQAgent` binds
  :func:`~rl_agents.double_q.double_q_loss`, and its own two bindings differ in the
  network alone.

The agent owns its network, its replay buffer and its epsilon schedule. It owns no
environment, no ``lax.scan`` and no discount: :func:`rl_components.loop.run` supplies the
horizon and ``gamma``, which is why no config here has a ``GAMMA`` field. Two sources of
truth for the discount is how the bootstrap defect this port exists to fix recurs.

Three things ``step`` needs are spec-derived, and ``step`` never sees the spec -- it
receives only a :class:`~rl_components.timestep.Timestep` -- while the agent object itself
is static under ``jit`` and so cannot be mutated by ``init``. Each is therefore reachable
from the state instead:

- the network, through ``state.algorithm_state.train_state.apply_fn``, which Flax marks
  ``pytree_node=False`` and which therefore survives the scan carry as a static field;
- the action count, as an int32 leaf, so ``jax.random.randint`` accepts a traced
  ``maxval``, nested in ``state.algorithm_state``;
- the replay buffer, rebuilt inside ``step`` from the shapes and dtypes of ``state.replay_state``.
  ``ReplayBuffer`` holds no data, only that geometry, so reconstructing it is free and
  needs nothing the state does not already carry.

``dqn_atari`` deliberately stays outside this class. Its ``can_train`` gates on replay
occupancy rather than on ``step_index``, and reconciling that convention is a behaviour
change, not a refactor; folding it in here would silently move one of the two agents onto
the other's schedule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_utils.pytree import tree_lerp
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.exploration import linear_epsilon_schedule
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

type QApplyFn = Callable[..., jax.Array]
"""A Flax module's ``apply``, as reached through ``TrainState.apply_fn``."""


class QNetworkModule(Protocol):
    """The two methods the shared implementation needs from a Q-network module."""

    def init(self, rngs: chex.PRNGKey, x: jax.Array) -> VariableDict: ...

    def apply(self, variables: object, x: jax.Array, *, rngs: object | None = None) -> jax.Array: ...


type QNetworkFactory = Callable[[int, tuple[int, ...]], QNetworkModule]
"""Builds the Q-network from the discrete action count and the observation shape."""


class QLossFn(Protocol):
    """The target rule, which is the only thing separating the agents built on this class.

    Every implementation is a module-level function rather than a closure inside its
    agent, so a test can gate the real target rule instead of a copy of it.
    """

    def __call__(
        self,
        params: VariableDict,
        target_params: VariableDict,
        obs: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_obs: jax.Array,
        discounts: jax.Array,
        *,
        apply_fn: QApplyFn,
    ) -> jax.Array: ...


class ReplayQConfig(Protocol):
    """The hyperparameters the shared implementation reads.

    Every public config satisfies this structurally while keeping its own field
    declarations, so each agent's defaults stay readable in one place.
    """

    @property
    def LEARNING_RATE(self) -> float: ...

    @property
    def BUFFER_SIZE(self) -> int: ...

    @property
    def BATCH_SIZE(self) -> int: ...

    @property
    def TOTAL_TIMESTEPS(self) -> int: ...

    @property
    def LEARNING_STARTS(self) -> int: ...

    @property
    def TRAIN_FREQUENCY(self) -> int: ...

    @property
    def TARGET_NETWORK_FREQUENCY(self) -> int: ...

    @property
    def TAU(self) -> float: ...

    @property
    def EPSILON_START(self) -> float: ...

    @property
    def EPSILON_END(self) -> float: ...

    @property
    def EPSILON_FRACTION(self) -> float: ...


@chex_struct(frozen=True)
class ReplayQHypers:
    """The hyperparameters ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. Everything here is
    read only in arithmetic or in a traced predicate, never to size an array
    or take a Python branch -- those stay on the config.
    """

    LEARNING_RATE: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    TARGET_NETWORK_FREQUENCY: jax.Array
    TAU: jax.Array
    EPSILON_START: jax.Array
    EPSILON_END: jax.Array
    EPSILON_FRACTION: jax.Array
    TOTAL_TIMESTEPS: jax.Array


def replay_q_hypers(config: ReplayQConfig):
    return ReplayQHypers(
        LEARNING_RATE=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.LEARNING_STARTS, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.TRAIN_FREQUENCY, jnp.int32),
        TARGET_NETWORK_FREQUENCY=jnp.asarray(config.TARGET_NETWORK_FREQUENCY, jnp.int32),
        TAU=jnp.asarray(config.TAU, jnp.float32),
        EPSILON_START=jnp.asarray(config.EPSILON_START, jnp.float32),
        EPSILON_END=jnp.asarray(config.EPSILON_END, jnp.float32),
        EPSILON_FRACTION=jnp.asarray(config.EPSILON_FRACTION, jnp.float32),
        TOTAL_TIMESTEPS=jnp.asarray(config.TOTAL_TIMESTEPS, jnp.float32),
    )


@chex_struct(frozen=True)
class ReplayQAlgorithmState:
    """Online and target networks plus swept hyperparameters for ReplayQ."""

    train_state: TrainState
    target_params: VariableDict
    num_actions: jax.Array
    hypers: ReplayQHypers


class ReplayQAgent:
    """Off-policy Q-learning from a replay buffer, with a target network and epsilon-greedy exploration.

    Not a public agent on its own: the ``network_factory`` and ``loss_fn`` handed to
    ``__init__`` decide which agent this is, and no subclass overrides anything else.
    """

    def __init__(
        self,
        config: ReplayQConfig,
        network_factory: QNetworkFactory,
        loss_fn: QLossFn,
    ) -> None:
        """Bind the configuration, the network and the target rule. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
            network_factory: Builds the Q-network once ``init`` knows the action count
                and the observation shape.
            loss_fn: The target rule this agent regresses towards.
        """
        self.config = config
        self.network_factory = network_factory
        self.loss_fn = loss_fn

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> ReplayAgentState[ReplayQAlgorithmState, ReplayBufferState]:
        """Build parameters, optimizer, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description, and the only place the observation shape
                comes from. Must be discrete: the target maximizes over the action space.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space.
        """
        if spec.num_actions is None:
            raise ValueError(f"Q-learning requires a discrete action space, got spec {spec.id!r} with none")

        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = self.network_factory(spec.num_actions, observation_shape)

        params_key, carry_key = jax.random.split(key)
        params = network.init(params_key, jnp.zeros(observation_shape, dtype=observation_dtype))
        buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=tuple(spec.action_shape),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        return init_replay_agent_state(
            ReplayQAlgorithmState(
                train_state=TrainState.create(
                    apply_fn=network.apply,
                    params=params,
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=jnp.asarray(self.config.LEARNING_RATE, jnp.float32)),
                ),
                target_params=params,
                num_actions=jnp.asarray(spec.num_actions, dtype=jnp.int32),
                hypers=replay_q_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: ReplayAgentState[ReplayQAlgorithmState, ReplayBufferState],
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[ReplayAgentState[ReplayQAlgorithmState, ReplayBufferState], jax.Array]:
        """Close the pending transition, learn from replay, sync the target, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so both
                insertion and learning are guarded above it.

        Returns:
            The next state, the action to apply, and a fixed three-key metric schema.
            ``loss`` is a zero placeholder on the iterations that do not learn, so the
            pytree returned to ``lax.scan`` is identical on every iteration.
        """
        algorithm_state = state.algorithm_state
        buffer = ReplayBuffer.from_state(state.replay_state)
        carry_key, sample_key, action_key = jax.random.split(state.key, 3)

        def close_transition(
            replay_state: ReplayBufferState,
            last_obs: jax.Array,
            last_action: jax.Array,
            current_timestep: Timestep[jax.Array],
        ) -> ReplayBufferState:
            return buffer.add(
                replay_state,
                last_obs[None, ...],
                last_action[None, ...],
                current_timestep.reward[None, ...],
                current_timestep.bootstrap_observation[None, ...],
                current_timestep.discount[None, ...],
            )

        def after_learn(learner_state: ReplayQAlgorithmState, current_step: jax.Array, _did_learn: jax.Array):
            return replace(
                learner_state,
                target_params=jax.lax.cond(
                    current_step % learner_state.hypers.TARGET_NETWORK_FREQUENCY == 0,
                    lambda: tree_lerp(
                        learner_state.target_params,
                        learner_state.train_state.params,
                        learner_state.hypers.TAU,
                    ),
                    lambda: learner_state.target_params,
                ),
            )

        def select_action(
            learner_state: ReplayQAlgorithmState,
            observation: jax.Array,
            current_step: jax.Array,
            keys: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            epsilon = linear_epsilon_schedule(
                current_step,
                learner_state.hypers.TOTAL_TIMESTEPS,
                learner_state.hypers.EPSILON_START,
                learner_state.hypers.EPSILON_END,
                learner_state.hypers.EPSILON_FRACTION,
            )
            q_values = jnp.asarray(learner_state.train_state.apply_fn(learner_state.train_state.params, observation))
            chose_random = jax.random.uniform(keys[0], ()) < epsilon
            random_action = jax.random.randint(keys[0], (), 0, learner_state.num_actions)
            action = jnp.where(chose_random, random_action, jnp.argmax(q_values)).astype(state.last_action.dtype)
            return action, {
                "epsilon": epsilon,
                "q_max": jnp.max(q_values).astype(jnp.float32),
            }

        hypers = algorithm_state.hypers
        return replay_agent_step(
            state,
            timestep,
            step_index,
            carry_key=carry_key,
            learn_keys=(sample_key,),
            action_keys=(action_key,),
            learning_starts=hypers.LEARNING_STARTS,
            train_frequency=hypers.TRAIN_FREQUENCY,
            warmup_steps=None,
            close_transition=close_transition,
            learn=partial(self._learn, buffer=buffer),
            select_action=select_action,
            after_learn=after_learn,
        )

    def _learn(
        self,
        learner_state: ReplayQAlgorithmState,
        replay_state: ReplayBufferState,
        keys: tuple[jax.Array, ...],
        *,
        buffer: ReplayBuffer,
    ) -> tuple[ReplayQAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
        """Take one gradient step on a replay minibatch.

        The rate is pushed into the optimizer state each step because the
        transformation itself is a static field and cannot hold a traced value.
        """
        train_state = with_hyperparams(learner_state.train_state, learning_rate=learner_state.hypers.LEARNING_RATE)
        obs, actions, rewards, next_obs, discounts = buffer.sample(replay_state, keys[0], self.config.BATCH_SIZE)

        def loss_fn(params: VariableDict) -> jax.Array:
            return self.loss_fn(
                params,
                learner_state.target_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                apply_fn=train_state.apply_fn,
            )

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        updated_state = replace(learner_state, train_state=train_state.apply_gradients(grads=grads))
        return updated_state, replay_state, {"loss": loss.astype(jnp.float32)}
