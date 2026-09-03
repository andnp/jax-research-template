"""DQN as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

The agent owns its networks, its replay buffer and its epsilon schedule. It owns no
environment, no ``lax.scan`` and no discount: :func:`rl_components.loop.run` supplies the
horizon and ``gamma``, which is why ``DQNConfig`` has no ``GAMMA`` field. Two sources of
truth for the discount is how the bootstrap defect this port exists to fix recurs.

Three things ``step`` needs are spec-derived, and ``step`` never sees the spec -- it
receives only a :class:`~rl_components.timestep.Timestep` -- while the agent object itself
is static under ``jit`` and so cannot be mutated by ``init``. Each is therefore reachable
from the state instead:

- the network, through ``state.train_state.apply_fn``, which Flax marks
  ``pytree_node=False`` and which therefore survives the scan carry as a static field;
- the action count, as an int32 leaf, so ``jax.random.randint`` accepts a traced
  ``maxval``;
- the replay buffer, rebuilt inside ``step`` from the shapes and dtypes of its own state.
  ``ReplayBuffer`` holds no data, only that geometry, so reconstructing it is free and
  needs nothing the state does not already carry.
"""

from __future__ import annotations

from typing import Literal

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

from rl_agents.q_networks import make_q_network


@chex_struct(frozen=True, kw_only=True)
class DQNConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    TARGET_NETWORK_FREQUENCY: int = 1_000
    TAU: float = 1.0  # Soft update
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42
    NETWORK_PRESET: Literal["mlp", "nature_cnn"] = "mlp"


@chex_struct(frozen=True)
class DQNAgentState:
    """Everything DQN carries between loop iterations.

    Attributes:
        train_state: Online parameters, optimizer state, and the static ``apply_fn``
            through which ``step`` reaches the Q-network.
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
    """

    train_state: TrainState
    target_params: VariableDict
    buffer_state: ReplayBufferState
    last_obs: jax.Array
    last_action: jax.Array
    num_actions: jax.Array
    key: chex.PRNGKey


def _buffer_from_state(buffer_state: ReplayBufferState) -> ReplayBuffer:
    """Rebuild the replay buffer from the geometry recorded in its own state."""
    return ReplayBuffer(
        capacity=buffer_state.obs.shape[0],
        obs_shape=buffer_state.obs.shape[1:],
        action_shape=buffer_state.actions.shape[1:],
        action_dtype=buffer_state.actions.dtype,
        obs_dtype=buffer_state.obs.dtype,
    )


class DQNAgent:
    """Deep Q-learning with a target network and epsilon-greedy exploration."""

    def __init__(self, config: DQNConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> DQNAgentState:
        """Build parameters, optimizer, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description. Must be discrete: DQN maximizes over the
                action space.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space.
        """
        if spec.num_actions is None:
            raise ValueError(f"DQN requires a discrete action space, got spec {spec.id!r} with none")

        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = make_q_network(self.config, spec.num_actions, observation_shape=observation_shape)

        params_key, carry_key = jax.random.split(key)
        params = network.init(params_key, jnp.zeros(observation_shape, dtype=observation_dtype))
        buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=tuple(spec.action_shape),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        return DQNAgentState(
            train_state=TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=optax.adam(self.config.LR),
            ),
            target_params=params,
            buffer_state=buffer.init(),
            last_obs=jnp.zeros(observation_shape, dtype=observation_dtype),
            last_action=jnp.zeros(tuple(spec.action_shape), dtype=action_dtype),
            num_actions=jnp.asarray(spec.num_actions, dtype=jnp.int32),
            key=carry_key,
        )

    def step(
        self,
        state: DQNAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[DQNAgentState, jax.Array]:
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
        config = self.config
        buffer = _buffer_from_state(state.buffer_state)
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

        can_train = (step_index > config.LEARNING_STARTS) & (step_index % config.TRAIN_FREQUENCY == 0)
        train_state, loss = jax.lax.cond(
            can_train,
            lambda: self._learn(state.train_state, state.target_params, buffer_state, buffer, sample_key),
            lambda: (state.train_state, jnp.zeros((), jnp.float32)),
        )

        target_params = jax.lax.cond(
            step_index % config.TARGET_NETWORK_FREQUENCY == 0,
            lambda: jax.tree_util.tree_map(
                lambda target, online: config.TAU * online + (1.0 - config.TAU) * target,
                state.target_params,
                train_state.params,
            ),
            lambda: state.target_params,
        )

        epsilon = jnp.maximum(
            jnp.asarray(config.EPSILON_END, jnp.float32),
            jnp.asarray(config.EPSILON_START, jnp.float32)
            - (config.EPSILON_START - config.EPSILON_END)
            * (step_index / (config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION)),
        ).astype(jnp.float32)
        q_values = jnp.asarray(train_state.apply_fn(train_state.params, timestep.observation))
        chose_random = jax.random.uniform(action_key, ()) < epsilon
        random_action = jax.random.randint(action_key, (), 0, state.num_actions)
        action = jnp.where(chose_random, random_action, jnp.argmax(q_values)).astype(state.last_action.dtype)

        return AgentStep(
            state=DQNAgentState(
                train_state=train_state,
                target_params=target_params,
                buffer_state=buffer_state,
                last_obs=timestep.observation,
                last_action=action,
                num_actions=state.num_actions,
                key=carry_key,
            ),
            action=action,
            metrics={
                "loss": loss,
                "epsilon": epsilon,
                "q_max": jnp.max(q_values).astype(jnp.float32),
            },
        )

    def _learn(
        self,
        train_state: TrainState,
        target_params: VariableDict,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
    ) -> tuple[TrainState, jax.Array]:
        """Take one gradient step on a replay minibatch."""
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, key, self.config.BATCH_SIZE)

        def _loss_fn(params: VariableDict) -> jax.Array:
            q_values = jnp.asarray(train_state.apply_fn(params, obs))
            q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()
            next_q_max = jnp.max(jnp.asarray(train_state.apply_fn(target_params, next_obs)), axis=-1)
            target = rewards + discounts * next_q_max
            return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))

        loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
        return train_state.apply_gradients(grads=grads), loss.astype(jnp.float32)
