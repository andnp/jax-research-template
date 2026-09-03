"""Soft Actor-Critic as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

:class:`SACAgent` is the port. ``make_train`` below it is the private training loop the
port replaces, and it is still here for one reason: ``projects/process-control-baselines``
calls ``rl_agents.sac.make_train`` from a different repository, and no commit spans two
repositories. **Removal condition:** once that project drives :class:`SACAgent` through
:func:`rl_components.loop.run`, ``make_train``, :class:`RunnerState`,
:class:`SACTrainOutput`, ``SACConfig.GAMMA`` and the
:class:`~rl_components.gym_env.GymEnv` import all go together.

``GAMMA`` survives only for that legacy path. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in replay, and
:class:`SACAgent` reads that coefficient; it never reads ``config.GAMMA``. Two sources of
truth for the discount is how the bootstrap defect this port exists to fix recurs.

The three losses are module-level functions rather than closures inside ``make_train``,
and that is the second thing the port buys. While the real expressions were unreachable
from outside this module, the tests that gated SAC re-implemented its arithmetic on local
arrays and so agreed with a copy of themselves rather than with the agent. Both paths now
call the same three functions.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple, TypedDict, cast

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.gym_env import ContinuousActionSpace, GymEnv
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

from rl_agents.continuous_actions import continuous_action_dim, uniform_exploration_action


@chex_struct(frozen=True, kw_only=True)
class SACConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 5_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99  # Legacy make_train only; see the module docstring.
    TAU: float = 0.005
    ALPHA: float = 0.2
    TARGET_ENTROPY: float | None = None
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


@chex_struct(frozen=True)
class SACHypers:
    """The hyperparameters ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. ``TARGET_ENTROPY`` is
    not here: it is resolved from ``config.TARGET_ENTROPY`` (or the action
    dimension) into a Python float at construction time, so it stays a
    per-state leaf derived once in ``init`` rather than a swept hyper.
    """

    LR: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    TAU: jax.Array


def sac_hypers(config: SACConfig) -> SACHypers:
    return SACHypers(
        LR=jnp.asarray(config.LR, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.LEARNING_STARTS, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.TRAIN_FREQUENCY, jnp.int32),
        TAU=jnp.asarray(config.TAU, jnp.float32),
    )


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


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
        std = jnp.exp(log_std)
        normal = distrax.Normal(mean, std)
        x_t = normal.sample(seed=rng)
        y_t = jnp.tanh(x_t)
        action = y_t

        # Log prob adjustment for Tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= jnp.log(1.0 - y_t**2 + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1)
        return action, log_prob


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
    return jax.vmap(
        lambda params, o, a: _critic_apply(critic, params, o, a),
        in_axes=(0, None, None),
    )(critic_params, obs, actions)


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
        key: PRNG key for the action sample.
        actor: The policy module.

    Returns:
        The scalar loss.
    """
    _, sample_key = jax.random.split(key)
    _, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, obs, jax.random.split(sample_key, obs.shape[0])
    )
    return -jnp.mean(cast(jax.Array, alpha_params["log_alpha"]) * (log_probs + target_entropy))


@chex_struct(frozen=True)
class SACAgentState:
    """Everything :class:`SACAgent` carries between loop iterations.

    Attributes:
        actor_state: Policy parameters, optimizer state and the static ``apply_fn``.
        critic_state: Twin-critic ensemble parameters, leading axis over members.
        critic_target_params: Target-network parameters the bootstrap value is read from,
            Polyak-averaged towards the online ensemble on every gradient step.
        alpha_state: The tuned entropy temperature, held as ``log_alpha``. Its
            ``apply_fn`` is ``None``: there is no network, only a parameter.
        buffer_state: Replay contents, and the only record of the buffer's geometry.
        last_obs: Observation the pending transition started from. Zero-primed at
            ``init``; never read before the first insertion, which is guarded on
            ``step_index > 0``.
        last_action: Action that opened the pending transition, zero-primed likewise. Its
            shape is also where ``step`` recovers the action dimension.
        target_entropy: The entropy target, as a float32 leaf. It is derived from the
            action dimension, which only ``init`` sees, so ``step`` reaches it here.
        key: PRNG key for sampling minibatches, actions and exploration.
        hypers: Swept hyperparameters, traced so a batch of arms shares one kernel.
    """

    actor_state: TrainState
    critic_state: TrainState
    critic_target_params: VariableDict
    alpha_state: TrainState
    buffer_state: ReplayBufferState
    last_obs: jax.Array
    last_action: jax.Array
    target_entropy: jax.Array
    key: chex.PRNGKey
    hypers: SACHypers


class SACAgent:
    """Maximum-entropy off-policy actor-critic over a continuous action space.

    The policy is tanh-squashed, so the environment's action space must already be
    normalized to ``[-1, 1]``; see :mod:`rl_agents.continuous_actions`.
    """

    def __init__(self, config: SACConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar. ``GAMMA`` is not read at all.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> SACAgentState:
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
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=(action_dim,),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )
        target_entropy = -float(action_dim) if self.config.TARGET_ENTROPY is None else self.config.TARGET_ENTROPY

        lr = jnp.asarray(self.config.LR, jnp.float32)
        return SACAgentState(
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
                params={"log_alpha": jnp.log(jnp.array([self.config.ALPHA], dtype=jnp.float32))},
                tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
            ),
            buffer_state=buffer.init(),
            last_obs=zero_obs,
            last_action=zero_action,
            target_entropy=jnp.asarray(target_entropy, dtype=jnp.float32),
            key=carry_key,
            hypers=sac_hypers(self.config),
        )

    def step(
        self,
        state: SACAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[SACAgentState, jax.Array]:
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
        action_dim = state.last_action.shape[0]
        actor, _ = _networks(action_dim)
        buffer = ReplayBuffer.from_state(state.buffer_state)
        carry_key, learn_key, action_key = jax.random.split(state.key, 3)

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
        can_train = (step_index > hypers.LEARNING_STARTS) & (step_index % hypers.TRAIN_FREQUENCY == 0)
        (
            actor_state,
            critic_state,
            critic_target_params,
            alpha_state,
            critic_loss,
            actor_loss,
            alpha_loss,
        ) = jax.lax.cond(
            can_train,
            lambda: self._learn(state, buffer_state, buffer, learn_key),
            lambda: (
                state.actor_state,
                state.critic_state,
                state.critic_target_params,
                state.alpha_state,
                jnp.zeros((), jnp.float32),
                jnp.zeros((), jnp.float32),
                jnp.zeros((), jnp.float32),
            ),
        )

        action = jax.lax.cond(
            step_index < hypers.LEARNING_STARTS,
            lambda: uniform_exploration_action(action_key, (action_dim,), state.last_action.dtype),
            lambda: actor.sample(actor_state.params, timestep.observation, action_key)[0].astype(
                state.last_action.dtype
            ),
        )

        return AgentStep(
            state=SACAgentState(
                actor_state=actor_state,
                critic_state=critic_state,
                critic_target_params=critic_target_params,
                alpha_state=alpha_state,
                buffer_state=buffer_state,
                last_obs=timestep.observation,
                last_action=action,
                target_entropy=state.target_entropy,
                key=carry_key,
                hypers=hypers,
            ),
            action=action,
            metrics={
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": jnp.exp(cast(jax.Array, alpha_state.params["log_alpha"])[0]).astype(jnp.float32),
            },
        )

    def _learn(
        self,
        state: SACAgentState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
    ) -> tuple[TrainState, TrainState, VariableDict, TrainState, jax.Array, jax.Array, jax.Array]:
        """Take one critic, actor and temperature gradient step on a replay minibatch.

        The three updates are sequential and each reads its predecessor's output, as in
        the reference algorithm: the actor is scored by the critic that has just moved,
        and the temperature is measured against the policy that has just moved.
        """
        hypers = state.hypers
        actor, critic = _networks(state.last_action.shape[0])
        sample_key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, sample_key, self.config.BATCH_SIZE)
        alpha = jnp.exp(cast(jax.Array, state.alpha_state.params["log_alpha"])[0])

        def _with_lr(train_state: TrainState) -> TrainState:
            opt_state = train_state.opt_state
            return train_state.replace(
                opt_state=opt_state._replace(hyperparams={**opt_state.hyperparams, "learning_rate": hypers.LR})
            )

        critic_state = _with_lr(state.critic_state)
        critic_loss, critic_grads = jax.value_and_grad(partial(sac_critic_loss, actor=actor, critic=critic))(
            critic_state.params,
            state.actor_state.params,
            state.critic_target_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            alpha,
            critic_key,
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        actor_state = _with_lr(state.actor_state)
        actor_loss, actor_grads = jax.value_and_grad(partial(sac_actor_loss, actor=actor, critic=critic))(
            actor_state.params, critic_state.params, obs, alpha, actor_key
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        alpha_state = _with_lr(state.alpha_state)
        alpha_loss, alpha_grads = jax.value_and_grad(partial(sac_alpha_loss, actor=actor))(
            alpha_state.params, actor_state.params, obs, state.target_entropy, alpha_key
        )
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        critic_target_params = jax.tree_util.tree_map(
            lambda target, online: hypers.TAU * online + (1.0 - hypers.TAU) * target,
            state.critic_target_params,
            critic_state.params,
        )

        return (
            actor_state,
            critic_state,
            critic_target_params,
            alpha_state,
            critic_loss.astype(jnp.float32),
            actor_loss.astype(jnp.float32),
            alpha_loss.astype(jnp.float32),
        )


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    critic_target_params: VariableDict
    alpha_state: TrainState
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class SACTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: SACConfig, env: GymEnv[ContinuousActionSpace], env_params: object | None = None) -> Callable[[jax.Array], SACTrainOutput]:
    def train(rng: jax.Array) -> SACTrainOutput:
        # INIT NETWORKS
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        action_dim = env.action_space(env_params).shape[0]
        obs_dim = env.observation_space(env_params).shape

        actor, critic = _networks(action_dim)
        actor_params = actor.init(_rng_actor, jnp.zeros(obs_dim))
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.LR))

        rng, _rng_critic = jax.random.split(rng)
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
            jax.random.split(_rng_critic, 2), jnp.zeros(obs_dim), jnp.zeros((action_dim,))
        )
        critic_target_params = critic_params
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.LR))

        # Automatic Entropy Tuning
        if config.TARGET_ENTROPY is None:
            target_entropy = -float(action_dim)
        else:
            target_entropy = config.TARGET_ENTROPY

        log_alpha = {"log_alpha": jnp.log(jnp.array([config.ALPHA]))}
        alpha_state = TrainState.create(apply_fn=None, params=log_alpha, tx=optax.adam(config.LR))

        # INIT BUFFER
        action_shape = env.action_space(env_params).shape
        buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, action_shape, jnp.float32)
        buffer_state = buffer.init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def _update_step(runner_state: RunnerState, t: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            (
                actor_state,
                critic_state,
                critic_target_params,
                alpha_state,
                buffer_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            def _random_action() -> jax.Array:
                return jax.random.uniform(_rng, (action_dim,), minval=-1, maxval=1)

            def _policy_action() -> jax.Array:
                action, _ = actor.sample(actor_state.params, last_obs, _rng)
                return action

            action = jax.lax.cond(
                t < config.LEARNING_STARTS,
                _random_action,
                _policy_action,
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            discount = config.GAMMA * (1.0 - done)

            # ADD TO BUFFER
            buffer_state = buffer.add(
                buffer_state,
                last_obs[None, ...],
                action[None, ...],
                reward[None, ...],
                obsv[None, ...],
                discount[None, ...],
            )

            # TRAIN
            def _do_train(
                actor_state: TrainState,
                critic_state: TrainState,
                critic_target_params: VariableDict,
                alpha_state: TrainState,
                buffer_state: ReplayBufferState,
                rng: jax.Array,
            ) -> tuple[TrainState, TrainState, VariableDict, TrainState, jax.Array, jax.Array]:
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

                log_alpha_arr = cast(jax.Array, alpha_state.params["log_alpha"])
                alpha = jnp.exp(log_alpha_arr[0])

                # CRITIC UPDATE
                critic_loss, critic_grads = jax.value_and_grad(partial(sac_critic_loss, actor=actor, critic=critic))(
                    critic_state.params, actor_state.params, critic_target_params, obs, actions, rewards, next_obs, discounts, alpha, rng
                )
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                # ACTOR UPDATE
                actor_loss, actor_grads = jax.value_and_grad(partial(sac_actor_loss, actor=actor, critic=critic))(
                    actor_state.params, critic_state.params, obs, alpha, rng
                )
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                # ALPHA UPDATE
                _, alpha_grads = jax.value_and_grad(partial(sac_alpha_loss, actor=actor))(
                    alpha_state.params, actor_state.params, obs, target_entropy, rng
                )
                alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

                # TARGET UPDATE
                critic_target_params = jax.tree_util.tree_map(
                    lambda tp, p: config.TAU * p + (1.0 - config.TAU) * tp,
                    critic_target_params,
                    critic_state.params,
                )

                return actor_state, critic_state, critic_target_params, alpha_state, critic_loss, actor_loss

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, critic_target_params, alpha_state, critic_loss, actor_loss = jax.lax.cond(
                can_train,
                lambda: _do_train(actor_state, critic_state, critic_target_params, alpha_state, buffer_state, rng),
                lambda: (actor_state, critic_state, critic_target_params, alpha_state, jnp.array(0.0), jnp.array(0.0)),
            )

            runner_state = RunnerState(
                actor_state=actor_state, critic_state=critic_state,
                critic_target_params=critic_target_params, alpha_state=alpha_state,
                buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng,
            )
            info = dict(info)
            info["critic_loss"] = critic_loss
            info["actor_loss"] = actor_loss
            return runner_state, info

        # RUNNER
        runner_state = RunnerState(
            actor_state=actor_state, critic_state=critic_state,
            critic_target_params=critic_target_params, alpha_state=alpha_state,
            buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
