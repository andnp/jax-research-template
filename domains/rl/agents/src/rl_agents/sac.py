"""Soft Actor-Critic as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

:class:`SACAgent` is the only training path here. The ``make_train`` closure that ran
beside it through the port is gone, and so is ``SACConfig.GAMMA``: no discount is
configurable on this agent any more. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in replay that
:class:`SACAgent` reads. Two sources of truth for the discount is how the bootstrap defect
this port exists to fix recurs.

The three losses are module-level functions rather than closures, which is the second
thing the port bought. While the real expressions were unreachable from outside this
module, the tests that gated SAC re-implemented its arithmetic on local arrays and so
agreed with a copy of themselves rather than with the agent. They call these functions now.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_utils.pytree import tree_lerp
from research_runner.config import traced
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.continuous_actions import continuous_action_dim
from rl_components.distributions import TanhNormalDiag
from rl_components.env_protocol import EnvSpec
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep
from rl_networks.networks import Critic


@chex_struct(frozen=True, kw_only=True)
class SACConfig:
    learning_rate: float = traced(3e-4)
    buffer_size: int = 100_000
    batch_size: int = 256
    total_timesteps: int = 1_000_000
    learning_starts: int = traced(5_000)
    train_frequency: int = traced(1)
    tau: float = traced(0.005)
    alpha: float = traced(0.2)
    target_entropy: float | None = None
    env_name: str = "MountainCarContinuous-v0"
    seed: int = 42

    def build(self) -> SACAgent:
        return SACAgent(self)


@chex_struct(frozen=True)
class SACHypers:
    """The hyperparameters ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. ``TARGET_ENTROPY`` is
    not here: it is resolved from ``config.target_entropy`` (or the action
    dimension) into a Python float at construction time, so it stays a
    per-state leaf derived once in ``init`` rather than a swept hyper.
    """

    LEARNING_RATE: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    TAU: jax.Array


def sac_hypers(config: SACConfig) -> SACHypers:
    return SACHypers(
        LEARNING_RATE=jnp.asarray(config.learning_rate, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.learning_starts, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.train_frequency, jnp.int32),
        TAU=jnp.asarray(config.tau, jnp.float32),
    )


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)
        return mean, log_std

    def sample(self, params: VariableDict, x: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean, log_std = _actor_apply(self, params, x)
        action, log_prob = TanhNormalDiag(mean=mean, log_std=log_std).sample_and_log_prob(rng)
        return action, jnp.sum(log_prob, axis=-1)


def _critic_apply(module: Critic, variables: VariableDict, x: jax.Array, a: jax.Array) -> jax.Array:
    return cast(jax.Array, module.apply(variables, x, a))


def _actor_apply(module: Actor, variables: VariableDict, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return cast(tuple[jax.Array, jax.Array], module.apply(variables, x))


def _networks(action_dim: int) -> tuple[Actor, Critic]:
    """Build the two Flax modules, which carry no state beyond the action dimension.

    ``init`` reads that dimension from the environment spec and ``step`` reads it back off
    the shape of the action it carries, so neither has to store the modules themselves.
    """
    return Actor(action_dim), Critic()


def _ensemble_q(critic: Critic, critic_params: VariableDict, obs: jax.Array, actions: jax.Array) -> jax.Array:
    """Every ensemble member's Q-value for one batch, shaped ``(members, batch)``."""

    def member_q(params: VariableDict, o: jax.Array, a: jax.Array) -> jax.Array:
        return _critic_apply(critic, params, o, a)

    return jax.vmap(member_q, in_axes=(0, None, None))(critic_params, obs, actions)


def sac_critic_loss(
    critic_params: VariableDict,
    actor_params: VariableDict,
    critic_target_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    alpha: jax.Array,
    key: chex.PRNGKey,
    *,
    actor: Actor,
    critic: Critic,
) -> jax.Array:
    """Mean squared error against the soft twin-Q target, over one replay minibatch.

    ``discounts`` carries the bootstrap coefficient the loop computed, so a terminal row's
    ``0.0`` removes the entire soft bootstrap -- both the twin-Q minimum and the entropy
    term -- and the loss becomes independent of ``next_obs`` on that row.

    Args:
        critic_params: Online parameters of the whole ensemble, leading axis over members.
            Differentiate with respect to this argument only.
        actor_params: Policy parameters, used to sample the bootstrap action.
        critic_target_params: Target-network parameters the bootstrap value is read from.
        obs: Observations the stored transitions started from.
        actions: Actions taken from ``obs``.
        rewards: Rewards the stored transitions earned.
        next_obs: True observations the stored transitions reached.
        discounts: Bootstrap coefficients, ``0.0`` wherever the transition terminated.
        alpha: Entropy temperature, treated as a constant here and tuned by
            :func:`sac_alpha_loss`.
        key: PRNG key for the bootstrap action sample.
        actor: The policy module, static because it is a Python object.
        critic: The critic module, likewise.

    Returns:
        The scalar loss, averaged over ensemble members.
    """
    _, sample_key = jax.random.split(key)
    next_actions, next_log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, next_obs, jax.random.split(sample_key, obs.shape[0])
    )

    next_q_min = jnp.min(_ensemble_q(critic, critic_target_params, next_obs, next_actions), axis=0)
    target_q = rewards + discounts * (next_q_min - alpha * next_log_probs)

    def _member_loss(params: VariableDict) -> jax.Array:
        q = _critic_apply(critic, params, obs, actions)
        return jnp.mean(jnp.square(q - jax.lax.stop_gradient(target_q)))

    return jnp.mean(jax.vmap(_member_loss)(critic_params))


def sac_actor_loss(
    actor_params: VariableDict,
    critic_params: VariableDict,
    obs: jax.Array,
    alpha: jax.Array,
    key: chex.PRNGKey,
    *,
    actor: Actor,
    critic: Critic,
) -> jax.Array:
    """Negative soft Q-value of freshly sampled actions, over one replay minibatch.

    Args:
        actor_params: Policy parameters. Differentiate with respect to this argument only.
        critic_params: Online critic ensemble, whose minimum scores the sampled actions.
        obs: Observations to act from.
        alpha: Entropy temperature.
        key: PRNG key for the action sample.
        actor: The policy module.
        critic: The critic module.

    Returns:
        The scalar loss.
    """
    _, sample_key = jax.random.split(key)
    new_actions, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, obs, jax.random.split(sample_key, obs.shape[0])
    )

    q_min = jnp.min(_ensemble_q(critic, critic_params, obs, new_actions), axis=0)
    return jnp.mean(alpha * log_probs - q_min)


def sac_alpha_loss(
    alpha_params: VariableDict,
    actor_params: VariableDict,
    obs: jax.Array,
    target_entropy: jax.Array | float,
    key: chex.PRNGKey,
    *,
    actor: Actor,
) -> jax.Array:
    """Automatic entropy tuning: drive the policy's entropy towards ``target_entropy``.

    Args:
        alpha_params: A single ``log_alpha`` leaf. Differentiate with respect to this
            argument only.
        actor_params: Policy parameters, whose log-probabilities are the measured entropy.
        obs: Observations to measure the policy's entropy at.
        target_entropy: The entropy the temperature is tuned to reach.
        key: Root key split into one action-sampling key per observation.
        actor: The policy module.

    Returns:
        The scalar loss.
    """
    _, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, obs, jax.random.split(key, obs.shape[0])
    )
    return -jnp.mean(cast(jax.Array, alpha_params["log_alpha"]) * (log_probs + target_entropy))


@chex_struct(frozen=True)
class SACAlgorithmState:
    """SAC learner state carried by the shared replay-agent wrapper."""

    actor_state: TrainState
    critic_state: TrainState
    critic_target_params: VariableDict
    alpha_state: TrainState
    target_entropy: jax.Array
    hypers: SACHypers


class SACAgent:
    """Maximum-entropy off-policy actor-critic over a continuous action space.

    The policy is tanh-squashed, so the environment's action space must already be
    normalized to ``[-1, 1]``; see :mod:`rl_components.continuous_actions`.
    """

    REPORTED_METRICS = ("critic_loss", "actor_loss", "alpha_loss", "alpha")
    INIT_SWEPT_FIELDS: tuple[str, ...] = ("alpha", "target_entropy")
    """Config fields ``init`` reads but ``step`` never traces -- see ``SACRCAgent``'s
    identical declaration, which this mirrors: ``init`` runs inside the vmapped region
    exactly as ``step`` does, so a per-arm traced value works there too.
    """

    def __init__(self, config: SACConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> ReplayAgentState[SACAlgorithmState, ReplayBufferState]:
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
        target_entropy = -float(action_dim) if self.config.target_entropy is None else self.config.target_entropy

        lr = jnp.asarray(self.config.learning_rate, jnp.float32)
        return init_replay_agent_state(
            SACAlgorithmState(
                actor_state=TrainState.create(
                    apply_fn=actor.apply,
                    params=actor.init(actor_key, zero_obs),
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                critic_state=TrainState.create(
                    apply_fn=critic.apply,
                    params=critic_params,
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                critic_target_params=critic_params,
                alpha_state=TrainState.create(
                    apply_fn=None,
                    params={"log_alpha": jnp.log(jnp.array([self.config.alpha], dtype=jnp.float32))},
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                target_entropy=jnp.asarray(target_entropy, dtype=jnp.float32),
                hypers=sac_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: ReplayAgentState[SACAlgorithmState, ReplayBufferState],
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[ReplayAgentState[SACAlgorithmState, ReplayBufferState], jax.Array]:
        """Close the pending transition, learn from replay, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes. The target
        network is Polyak-averaged inside the learning branch, which is where SAC's soft
        update belongs -- it is once per gradient step, not on a frequency of its own.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so both
                insertion and learning are guarded above it.

        Returns:
            The next state, the action to apply, and a fixed four-key metric schema. Each
            loss is a zero placeholder on the iterations that do not learn, so the pytree
            returned to ``lax.scan`` is identical on every iteration.
        """
        algorithm_state = state.algorithm_state
        action_dim = state.last_action.shape[0]
        actor, _ = _networks(action_dim)
        buffer = ReplayBuffer.from_state(state.replay_state)
        carry_key, learn_key, action_key = jax.random.split(state.key, 3)

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
            learner_state: SACAlgorithmState,
            replay_state: ReplayBufferState,
            keys: tuple[jax.Array, ...],
        ) -> tuple[SACAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
            return self._learn(learner_state, replay_state, buffer, keys[0], action_dim)

        def select_action(
            learner_state: SACAlgorithmState,
            observation: jax.Array,
            _step_index: jax.Array,
            keys: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            action = actor.sample(learner_state.actor_state.params, observation, keys[0])[0].astype(
                state.last_action.dtype
            )
            return action, {}

        hypers = algorithm_state.hypers
        return replay_agent_step(
            state,
            timestep,
            step_index,
            carry_key=carry_key,
            learn_keys=(learn_key,),
            action_keys=(action_key,),
            learning_starts=hypers.LEARNING_STARTS,
            train_frequency=hypers.TRAIN_FREQUENCY,
            warmup_steps=hypers.LEARNING_STARTS,
            close_transition=close_transition,
            learn=learn,
            select_action=select_action,
            metrics_when_not_learning={
                "alpha": jnp.exp(cast(jax.Array, algorithm_state.alpha_state.params["log_alpha"])[0]).astype(
                    jnp.float32
                )
            },
        )

    def _learn(
        self,
        learner_state: SACAlgorithmState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        action_dim: int,
    ) -> tuple[SACAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
        """Take one critic, actor and temperature gradient step on a replay minibatch.

        The three updates are sequential and each reads its predecessor's output, as in
        the reference algorithm: the actor is scored by the critic that has just moved,
        and the temperature is measured against the policy that has just moved.
        """
        hypers = learner_state.hypers
        actor, critic = _networks(action_dim)
        sample_key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        _, alpha_sample_key = jax.random.split(alpha_key)
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, sample_key, self.config.batch_size)
        alpha = jnp.exp(cast(jax.Array, learner_state.alpha_state.params["log_alpha"])[0])

        critic_state = with_hyperparams(learner_state.critic_state, learning_rate=hypers.LEARNING_RATE)
        critic_loss, critic_grads = jax.value_and_grad(partial(sac_critic_loss, actor=actor, critic=critic))(
            critic_state.params,
            learner_state.actor_state.params,
            learner_state.critic_target_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            alpha,
            critic_key,
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        actor_state = with_hyperparams(learner_state.actor_state, learning_rate=hypers.LEARNING_RATE)
        actor_loss, actor_grads = jax.value_and_grad(partial(sac_actor_loss, actor=actor, critic=critic))(
            actor_state.params, critic_state.params, obs, alpha, actor_key
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        alpha_state = with_hyperparams(learner_state.alpha_state, learning_rate=hypers.LEARNING_RATE)
        alpha_loss, alpha_grads = jax.value_and_grad(partial(sac_alpha_loss, actor=actor))(
            alpha_state.params, actor_state.params, obs, learner_state.target_entropy, alpha_sample_key
        )
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        updated_state = replace(
            learner_state,
            actor_state=actor_state,
            critic_state=critic_state,
            critic_target_params=tree_lerp(learner_state.critic_target_params, critic_state.params, hypers.TAU),
            alpha_state=alpha_state,
        )
        return updated_state, buffer_state, {
            "critic_loss": critic_loss.astype(jnp.float32),
            "actor_loss": actor_loss.astype(jnp.float32),
            "alpha_loss": alpha_loss.astype(jnp.float32),
            "alpha": jnp.exp(cast(jax.Array, alpha_state.params["log_alpha"])[0]).astype(jnp.float32),
        }
