"""TD3 as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

:class:`TD3Agent` is the whole module. The discount is absent from :class:`TD3Config`
because it belongs to :func:`rl_components.loop.run`, which stores a per-transition
coefficient in replay that the agent reads back; two sources of truth for it is the
bootstrap bug class the loop contract exists to prevent.

The three ideas TD3 adds to a deterministic actor-critic all live in the learning branch:
clipped double-Q and target policy smoothing in :func:`td3_critic_loss`, and the delayed
policy update in :meth:`TD3Agent._learn`. The delay covers *both* Polyak updates as well
as the actor step, so with ``policy_delay > 1`` the target networks stay stationary
between actor updates.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from jax_utils.pytree import tree_lerp
from research_runner.config import traced
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.continuous_actions import continuous_action_dim
from rl_components.env_protocol import EnvSpec
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep
from rl_networks.networks import Critic


@chex_struct(frozen=True, kw_only=True)
class TD3Config:
    learning_rate: float = traced(3e-4)
    buffer_size: int = 100_000
    batch_size: int = 256
    total_timesteps: int = 1_000_000
    learning_starts: int = traced(25_000)
    train_frequency: int = traced(1)
    tau: float = traced(0.005)
    policy_delay: int = traced(2)
    exploration_noise: float = traced(0.1)
    policy_noise: float = traced(0.2)
    noise_clip: float = traced(0.5)
    env_name: str = "MountainCarContinuous-v0"
    seed: int = 42

    def build(self) -> TD3Agent:
        return TD3Agent(self)


@chex_struct(frozen=True)
class TD3Hypers:
    """The hyperparameters ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. ``POLICY_DELAY`` is
    here despite reading like a structural period: it is only a traced modulo
    comparand feeding ``lax.cond``, so the compiled graph is identical
    regardless of its value.
    """

    LEARNING_RATE: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    TAU: jax.Array
    POLICY_DELAY: jax.Array
    EXPLORATION_NOISE: jax.Array
    POLICY_NOISE: jax.Array
    NOISE_CLIP: jax.Array


def td3_hypers(config: TD3Config) -> TD3Hypers:
    return TD3Hypers(
        LEARNING_RATE=jnp.asarray(config.learning_rate, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.learning_starts, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.train_frequency, jnp.int32),
        TAU=jnp.asarray(config.tau, jnp.float32),
        POLICY_DELAY=jnp.asarray(config.policy_delay, jnp.int32),
        EXPLORATION_NOISE=jnp.asarray(config.exploration_noise, jnp.float32),
        POLICY_NOISE=jnp.asarray(config.policy_noise, jnp.float32),
        NOISE_CLIP=jnp.asarray(config.noise_clip, jnp.float32),
    )


class Actor(TypedApply[jax.Array], nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return jnp.tanh(x)


def _soft_update(tau: jax.Array | float, target: VariableDict, online: VariableDict) -> VariableDict:
    return tree_lerp(target, online, tau)


def td3_critic_loss(
    critic_params: VariableDict,
    actor_target_params: VariableDict,
    critic_target_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    rng: jax.Array,
    policy_noise: jax.Array,
    noise_clip: jax.Array,
    *,
    actor: Actor,
    critic: Critic,
) -> jax.Array:
    """Mean squared error against the smoothed, clipped double-Q target, over one minibatch.

    ``discounts`` carries the bootstrap coefficient the loop computed, so a terminal row's
    ``0.0`` removes the bootstrap term outright and the loss becomes independent of
    ``next_obs`` on that row.

    Args:
        critic_params: Online parameters of the whole ensemble, leading axis over members.
            Differentiate with respect to this argument only.
        actor_target_params: Target policy, which proposes the bootstrap action.
        critic_target_params: Target critics, whose minimum values that action -- the
            clipping that keeps TD3's overestimation bounded.
        obs: Observations the stored transitions started from.
        actions: Actions taken from ``obs``.
        rewards: Rewards the stored transitions earned.
        next_obs: True observations the stored transitions reached.
        discounts: Bootstrap coefficients, ``0.0`` wherever the transition terminated.
        rng: PRNG key for target policy smoothing.
        policy_noise: Std of the target-policy smoothing noise.
        noise_clip: Clip bound applied to that noise.
        actor: The policy module, static because it is a Python object.
        critic: The critic module, likewise.

    Returns:
        The scalar loss, averaged over ensemble members.
    """
    rng, _rng_noise = jax.random.split(rng)
    next_actions = actor.apply(actor_target_params, next_obs)
    noise = jax.random.normal(_rng_noise, next_actions.shape) * policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    next_q = jax.vmap(
        critic.apply,
        in_axes=(0, None, None),
    )(critic_target_params, next_obs, next_actions)
    target_q = rewards + discounts * jnp.min(next_q, axis=0)

    def _single(p: VariableDict) -> jax.Array:
        q = critic.apply(p, obs, actions)
        return jnp.mean(jnp.square(q - jax.lax.stop_gradient(target_q)))

    return jnp.mean(jax.vmap(_single)(critic_params))


def td3_actor_loss(
    actor_params: VariableDict,
    critic_params: VariableDict,
    obs: jax.Array,
    *,
    actor: Actor,
    critic: Critic,
) -> jax.Array:
    """Negative Q-value of the deterministic policy's own actions, under the first critic.

    Only member ``0`` scores the actor, as in the reference algorithm: the clipped minimum
    exists to bound the critic's own target, not to pessimise the policy gradient.

    Args:
        actor_params: Policy parameters. Differentiate with respect to this argument only.
        critic_params: Online critic ensemble.
        obs: Observations to act from.
        actor: The policy module.
        critic: The critic module.

    Returns:
        The scalar loss.
    """
    new_actions = actor.apply(actor_params, obs)

    def member_q(p: VariableDict, o: jax.Array, a: jax.Array) -> jax.Array:
        return critic.apply(p, o, a)

    q_values = jax.vmap(member_q, in_axes=(0, None, None))(critic_params, obs, new_actions)
    return -jnp.mean(q_values[0])


def _networks(action_dim: int) -> tuple[Actor, Critic]:
    """Build the two Flax modules, which carry no state beyond the action dimension.

    ``init`` reads that dimension from the environment spec and ``step`` reads it back off
    the shape of the action it carries, so neither has to store the modules themselves.
    """
    return Actor(action_dim), Critic()


@chex_struct(frozen=True)
class TD3AlgorithmState:
    """TD3 learner state carried by the shared replay-agent wrapper."""

    actor_state: TrainState
    critic_state: TrainState
    actor_target_params: VariableDict
    critic_target_params: VariableDict
    hypers: TD3Hypers


class TD3Agent:
    """Twin Delayed DDPG: a deterministic policy, clipped double-Q, and Gaussian exploration.

    The policy is tanh-squashed, so the environment's action space must already be
    normalized to ``[-1, 1]``; see :mod:`rl_components.continuous_actions`.
    """

    REPORTED_METRICS = ("critic_loss", "actor_loss")
    INIT_SWEPT_FIELDS: tuple[str, ...] = ()
    """Config fields ``init`` reads that ``step`` never traces. None here."""

    def __init__(self, config: TD3Config) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> ReplayAgentState[TD3AlgorithmState, ReplayBufferState]:
        """Build networks, optimizers, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description, and the only place the observation shape
                and action dimension come from.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` is not a one-dimensional continuous action space
                normalized to ``[-1, 1]``.
        """
        action_dim = continuous_action_dim(spec)
        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        actor, critic = _networks(action_dim)

        actor_key, critic_key, carry_key = jax.random.split(key, 3)
        zero_obs = jnp.zeros(observation_shape, dtype=observation_dtype)
        zero_action = jnp.zeros((action_dim,), dtype=action_dtype)
        actor_params = actor.init(actor_key, zero_obs)
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
            jax.random.split(critic_key, 2), zero_obs, zero_action
        )
        buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            obs_shape=observation_shape,
            action_shape=(action_dim,),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        lr = jnp.asarray(self.config.learning_rate, jnp.float32)
        return init_replay_agent_state(
            TD3AlgorithmState(
                actor_state=TrainState.create(
                    apply_fn=actor.apply,
                    params=actor_params,
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                critic_state=TrainState.create(
                    apply_fn=critic.apply,
                    params=critic_params,
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                actor_target_params=actor_params,
                critic_target_params=critic_params,
                hypers=td3_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: ReplayAgentState[TD3AlgorithmState, ReplayBufferState],
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[ReplayAgentState[TD3AlgorithmState, ReplayBufferState], jax.Array]:
        """Close the pending transition, learn from replay, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes. Both
        Polyak updates happen inside the learning branch, delayed with the actor.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so both
                insertion and learning are guarded above it.

        Returns:
            The next state, the action to apply, and a fixed two-key metric schema. Each
            loss is a zero placeholder on the iterations that do not produce it --
            ``actor_loss`` also on the iterations the policy delay skips -- so the pytree
            returned to ``lax.scan`` is identical on every iteration.
        """
        algorithm_state = state.algorithm_state
        action_dim = state.last_action.shape[0]
        actor, _ = _networks(action_dim)
        buffer = ReplayBuffer.from_state(state.replay_state)
        carry_key, learn_key, action_key, noise_key = jax.random.split(state.key, 4)

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

        def learn(
            learner_state: TD3AlgorithmState,
            replay_state: ReplayBufferState,
            keys: tuple[jax.Array, ...],
        ) -> tuple[TD3AlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
            return self._learn(learner_state, replay_state, buffer, keys[0], step_index, action_dim)

        def select_action(
            learner_state: TD3AlgorithmState,
            observation: jax.Array,
            _step_index: jax.Array,
            keys: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            action = actor.apply(learner_state.actor_state.params, observation)
            noise = jax.random.normal(keys[1], action.shape) * learner_state.hypers.EXPLORATION_NOISE
            action = jnp.clip(action + noise, -1.0, 1.0).astype(state.last_action.dtype)
            return action, {}

        hypers = algorithm_state.hypers
        return replay_agent_step(
            state,
            timestep,
            step_index,
            carry_key=carry_key,
            learn_keys=(learn_key,),
            action_keys=(action_key, noise_key),
            learning_starts=hypers.LEARNING_STARTS,
            train_frequency=hypers.TRAIN_FREQUENCY,
            warmup_steps=hypers.LEARNING_STARTS,
            close_transition=close_transition,
            learn=learn,
            select_action=select_action,
        )

    def _learn(
        self,
        learner_state: TD3AlgorithmState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        step_index: jax.Array,
        action_dim: int,
    ) -> tuple[TD3AlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
        """Take a critic gradient step, and an actor step plus both Polyak updates on the delay."""
        hypers = learner_state.hypers
        actor, critic = _networks(action_dim)
        sample_key, critic_key = jax.random.split(key)
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, sample_key, self.config.batch_size)

        critic_state = with_hyperparams(learner_state.critic_state, learning_rate=hypers.LEARNING_RATE)
        critic_loss, critic_grads = jax.value_and_grad(partial(td3_critic_loss, actor=actor, critic=critic))(
            critic_state.params,
            learner_state.actor_target_params,
            learner_state.critic_target_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            critic_key,
            hypers.POLICY_NOISE,
            hypers.NOISE_CLIP,
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        def _update_actor() -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
            actor_state = with_hyperparams(learner_state.actor_state, learning_rate=hypers.LEARNING_RATE)
            actor_loss, actor_grads = jax.value_and_grad(partial(td3_actor_loss, actor=actor, critic=critic))(
                actor_state.params, critic_state.params, obs
            )
            actor_state = actor_state.apply_gradients(grads=actor_grads)
            return (
                actor_state,
                _soft_update(hypers.TAU, learner_state.actor_target_params, actor_state.params),
                _soft_update(hypers.TAU, learner_state.critic_target_params, critic_state.params),
                actor_loss.astype(jnp.float32),
            )

        def _skip_actor() -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
            return (
                learner_state.actor_state,
                learner_state.actor_target_params,
                learner_state.critic_target_params,
                jnp.zeros((), jnp.float32),
            )

        actor_state, actor_target_params, critic_target_params, actor_loss = jax.lax.cond(
            step_index % hypers.POLICY_DELAY == 0, _update_actor, _skip_actor
        )

        updated_state = replace(
            learner_state,
            actor_state=actor_state,
            critic_state=critic_state,
            actor_target_params=actor_target_params,
            critic_target_params=critic_target_params,
        )
        return updated_state, buffer_state, {
            "critic_loss": critic_loss.astype(jnp.float32),
            "actor_loss": actor_loss,
        }
