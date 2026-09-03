"""TD3 as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

:class:`TD3Agent` is the port. ``make_train`` below it is the private training loop the
port replaces, and it is still here for one reason: ``projects/process-control-baselines``
calls ``rl_agents.td3.make_train`` from a different repository, and no commit spans two
repositories. **Removal condition:** once that project drives :class:`TD3Agent` through
:func:`rl_components.loop.run`, ``make_train``, :class:`RunnerState`,
:class:`TD3TrainOutput`, ``TD3Config.GAMMA`` and the
:class:`~rl_components.gym_env.GymEnv` import all go together.

``GAMMA`` survives only for that legacy path. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in replay, and
:class:`TD3Agent` reads that coefficient; it never reads ``config.GAMMA``.

The three ideas TD3 adds to a deterministic actor-critic all live in the learning branch:
clipped double-Q and target policy smoothing in :func:`td3_critic_loss`, and the delayed
policy update in :meth:`TD3Agent._learn`. The delay covers *both* Polyak updates as well
as the actor step, so with ``POLICY_DELAY > 1`` the target networks stay stationary
between actor updates.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple, TypedDict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.gym_env import ContinuousActionSpace, GymEnv
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

from rl_agents.continuous_actions import continuous_action_dim, uniform_exploration_action


@chex_struct(frozen=True, kw_only=True)
class TD3Config:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 25_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99  # Legacy make_train only; see the module docstring.
    TAU: float = 0.005
    POLICY_DELAY: int = 2
    EXPLORATION_NOISE: float = 0.1
    POLICY_NOISE: float = 0.2
    NOISE_CLIP: float = 0.5
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


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

    def apply(
        self,
        variables: object,
        x: jax.Array,
        a: jax.Array,
        *,
        rngs: object | None = None,
    ) -> jax.Array:
        return super().apply(variables, x, a, rngs=rngs)


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


def _soft_update(tau: float, target: VariableDict, online: VariableDict) -> VariableDict:
    return jax.tree_util.tree_map(lambda tp, p: tau * p + (1.0 - tau) * tp, target, online)


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
    *,
    actor: Actor,
    critic: Critic,
    config: TD3Config,
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
        actor: The policy module, static because it is a Python object.
        critic: The critic module, likewise.
        config: Hyperparameters, read for the smoothing noise scale and its clip.

    Returns:
        The scalar loss, averaged over ensemble members.
    """
    rng, _rng_noise = jax.random.split(rng)
    next_actions = actor.apply(actor_target_params, next_obs)
    noise = jax.random.normal(_rng_noise, next_actions.shape) * config.POLICY_NOISE
    noise = jnp.clip(noise, -config.NOISE_CLIP, config.NOISE_CLIP)
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
    q_values = jax.vmap(
        lambda p, o, a: critic.apply(p, o, a),
        in_axes=(0, None, None),
    )(critic_params, obs, new_actions)
    return -jnp.mean(q_values[0])


def _networks(action_dim: int) -> tuple[Actor, Critic]:
    """Build the two Flax modules, which carry no state beyond the action dimension.

    ``init`` reads that dimension from the environment spec and ``step`` reads it back off
    the shape of the action it carries, so neither has to store the modules themselves.
    """
    return Actor(action_dim), Critic()


@chex_struct(frozen=True)
class TD3AgentState:
    """Everything :class:`TD3Agent` carries between loop iterations.

    Attributes:
        actor_state: Deterministic policy parameters, optimizer state and the static
            ``apply_fn``.
        critic_state: Critic ensemble parameters, leading axis over members.
        actor_target_params: Target policy, which proposes the bootstrap action.
        critic_target_params: Target critics, which value it.
        buffer_state: Replay contents, and the only record of the buffer's geometry.
        last_obs: Observation the pending transition started from. Zero-primed at
            ``init``; never read before the first insertion, which is guarded on
            ``step_index > 0``.
        last_action: Action that opened the pending transition, zero-primed likewise. Its
            shape is also where ``step`` recovers the action dimension.
        key: PRNG key for sampling minibatches, smoothing targets and exploring.
    """

    actor_state: TrainState
    critic_state: TrainState
    actor_target_params: VariableDict
    critic_target_params: VariableDict
    buffer_state: ReplayBufferState
    last_obs: jax.Array
    last_action: jax.Array
    key: chex.PRNGKey


class TD3Agent:
    """Twin Delayed DDPG: a deterministic policy, clipped double-Q, and Gaussian exploration.

    The policy is tanh-squashed, so the environment's action space must already be
    normalized to ``[-1, 1]``; see :mod:`rl_agents.continuous_actions`.
    """

    def __init__(self, config: TD3Config) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar. ``GAMMA`` is not read at all.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> TD3AgentState:
        """Build networks, optimizers, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description, and the only place the observation shape
                and action dimension come from.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` does not describe a one-dimensional continuous action.
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
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=(action_dim,),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        return TD3AgentState(
            actor_state=TrainState.create(
                apply_fn=actor.apply, params=actor_params, tx=optax.adam(self.config.LR)
            ),
            critic_state=TrainState.create(
                apply_fn=critic.apply, params=critic_params, tx=optax.adam(self.config.LR)
            ),
            actor_target_params=actor_params,
            critic_target_params=critic_params,
            buffer_state=buffer.init(),
            last_obs=zero_obs,
            last_action=zero_action,
            key=carry_key,
        )

    def step(
        self,
        state: TD3AgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[TD3AgentState, jax.Array]:
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
        config = self.config
        action_dim = state.last_action.shape[0]
        actor, _ = _networks(action_dim)
        buffer = ReplayBuffer.from_state(state.buffer_state)
        carry_key, learn_key, action_key, noise_key = jax.random.split(state.key, 4)

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
        (
            actor_state,
            critic_state,
            actor_target_params,
            critic_target_params,
            critic_loss,
            actor_loss,
        ) = jax.lax.cond(
            can_train,
            lambda: self._learn(state, buffer_state, buffer, learn_key, step_index),
            lambda: (
                state.actor_state,
                state.critic_state,
                state.actor_target_params,
                state.critic_target_params,
                jnp.zeros((), jnp.float32),
                jnp.zeros((), jnp.float32),
            ),
        )

        def _explore() -> jax.Array:
            return uniform_exploration_action(action_key, (action_dim,), state.last_action.dtype)

        def _act() -> jax.Array:
            action = actor.apply(actor_state.params, timestep.observation)
            noise = jax.random.normal(noise_key, action.shape) * config.EXPLORATION_NOISE
            return jnp.clip(action + noise, -1.0, 1.0).astype(state.last_action.dtype)

        action = jax.lax.cond(step_index < config.LEARNING_STARTS, _explore, _act)

        return AgentStep(
            state=TD3AgentState(
                actor_state=actor_state,
                critic_state=critic_state,
                actor_target_params=actor_target_params,
                critic_target_params=critic_target_params,
                buffer_state=buffer_state,
                last_obs=timestep.observation,
                last_action=action,
                key=carry_key,
            ),
            action=action,
            metrics={"critic_loss": critic_loss, "actor_loss": actor_loss},
        )

    def _learn(
        self,
        state: TD3AgentState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        step_index: jax.Array,
    ) -> tuple[TrainState, TrainState, VariableDict, VariableDict, jax.Array, jax.Array]:
        """Take a critic gradient step, and an actor step plus both Polyak updates on the delay."""
        config = self.config
        actor, critic = _networks(state.last_action.shape[0])
        sample_key, critic_key = jax.random.split(key)
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, sample_key, config.BATCH_SIZE)

        critic_loss, critic_grads = jax.value_and_grad(
            partial(td3_critic_loss, actor=actor, critic=critic, config=config)
        )(
            state.critic_state.params,
            state.actor_target_params,
            state.critic_target_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            critic_key,
        )
        critic_state = state.critic_state.apply_gradients(grads=critic_grads)

        def _update_actor() -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
            actor_loss, actor_grads = jax.value_and_grad(partial(td3_actor_loss, actor=actor, critic=critic))(
                state.actor_state.params, critic_state.params, obs
            )
            actor_state = state.actor_state.apply_gradients(grads=actor_grads)
            return (
                actor_state,
                _soft_update(config.TAU, state.actor_target_params, actor_state.params),
                _soft_update(config.TAU, state.critic_target_params, critic_state.params),
                actor_loss.astype(jnp.float32),
            )

        def _skip_actor() -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
            return (
                state.actor_state,
                state.actor_target_params,
                state.critic_target_params,
                jnp.zeros((), jnp.float32),
            )

        actor_state, actor_target_params, critic_target_params, actor_loss = jax.lax.cond(
            step_index % config.POLICY_DELAY == 0, _update_actor, _skip_actor
        )

        return (
            actor_state,
            critic_state,
            actor_target_params,
            critic_target_params,
            critic_loss.astype(jnp.float32),
            actor_loss,
        )


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    critic_target_params: VariableDict
    actor_target_params: VariableDict
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class TD3TrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: TD3Config, env: GymEnv[ContinuousActionSpace], env_params: object | None = None) -> Callable[[jax.Array], TD3TrainOutput]:
    action_dim = env.action_space(env_params).shape[0]
    obs_dim = env.observation_space(env_params).shape
    action_shape = env.action_space(env_params).shape

    actor, critic = _networks(action_dim)
    buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, action_shape, jnp.float32)

    _bound_critic_loss = partial(td3_critic_loss, actor=actor, critic=critic, config=config)
    _bound_actor_loss = partial(td3_actor_loss, actor=actor, critic=critic)

    def train(rng: jax.Array) -> TD3TrainOutput:
        # INIT NETWORKS
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        actor_params = actor.init(_rng_actor, jnp.zeros(obs_dim))
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.LR))

        rng, _rng_critic = jax.random.split(rng)
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(jax.random.split(_rng_critic, 2), jnp.zeros(obs_dim), jnp.zeros((action_dim,)))
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.LR))
        critic_target_params = critic_params
        actor_target_params = actor_params

        # INIT BUFFER & ENV
        buffer_state = buffer.init()
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def _do_train(
            actor_state: TrainState,
            critic_state: TrainState,
            critic_target_params: VariableDict,
            actor_target_params: VariableDict,
            buffer_state: ReplayBufferState,
            rng: jax.Array,
            t: jax.Array,
        ) -> tuple[TrainState, TrainState, VariableDict, VariableDict, jax.Array, jax.Array]:
            rng, _rng, _rng_cl = jax.random.split(rng, 3)
            obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

            critic_loss, critic_grads = jax.value_and_grad(_bound_critic_loss)(
                critic_state.params,
                actor_target_params,
                critic_target_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                _rng_cl,
            )
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            def _update_actor(
                actor_state: TrainState,
                critic_state: TrainState,
                critic_target_params: VariableDict,
                actor_target_params: VariableDict,
                obs: jax.Array,
            ) -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
                actor_loss, actor_grads = jax.value_and_grad(_bound_actor_loss)(actor_state.params, critic_state.params, obs)
                actor_state = actor_state.apply_gradients(grads=actor_grads)
                return (
                    actor_state,
                    _soft_update(config.TAU, critic_target_params, critic_state.params),
                    _soft_update(config.TAU, actor_target_params, actor_state.params),
                    actor_loss,
                )

            def _skip_actor(
                actor_state: TrainState,
                critic_state: TrainState,
                critic_target_params: VariableDict,
                actor_target_params: VariableDict,
                obs: jax.Array,
            ) -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
                return actor_state, critic_target_params, actor_target_params, jnp.array(0.0)

            actor_state, critic_target_params, actor_target_params, actor_loss = jax.lax.cond(
                t % config.POLICY_DELAY == 0,
                _update_actor,
                _skip_actor,
                actor_state,
                critic_state,
                critic_target_params,
                actor_target_params,
                obs,
            )
            return actor_state, critic_state, critic_target_params, actor_target_params, critic_loss, actor_loss

        def _skip_train(
            actor_state: TrainState,
            critic_state: TrainState,
            critic_target_params: VariableDict,
            actor_target_params: VariableDict,
            buffer_state: ReplayBufferState,
            rng: jax.Array,
            t: jax.Array,
        ) -> tuple[TrainState, TrainState, VariableDict, VariableDict, jax.Array, jax.Array]:
            return actor_state, critic_state, critic_target_params, actor_target_params, jnp.array(0.0), jnp.array(0.0)

        def _update_step(runner_state: RunnerState, t: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            actor_state, critic_state, critic_target_params, actor_target_params, buffer_state, env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng_action, _rng_noise = jax.random.split(rng, 3)

            def _random_action() -> jax.Array:
                return jax.random.uniform(_rng_action, (action_dim,), minval=-1, maxval=1)

            def _policy_action() -> jax.Array:
                action = actor.apply(actor_state.params, last_obs)
                noise = jax.random.normal(_rng_noise, action.shape) * config.EXPLORATION_NOISE
                return jnp.clip(action + noise, -1.0, 1.0)

            action = jax.lax.cond(t < config.LEARNING_STARTS, _random_action, _policy_action)

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
            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, critic_target_params, actor_target_params, critic_loss, actor_loss = jax.lax.cond(
                can_train,
                _do_train,
                _skip_train,
                actor_state,
                critic_state,
                critic_target_params,
                actor_target_params,
                buffer_state,
                rng,
                t,
            )

            runner_state = RunnerState(
                actor_state=actor_state,
                critic_state=critic_state,
                critic_target_params=critic_target_params,
                actor_target_params=actor_target_params,
                buffer_state=buffer_state,
                env_state=env_state,
                last_obs=obsv,
                rng=rng,
            )
            info = dict(info)
            info["critic_loss"] = critic_loss
            info["actor_loss"] = actor_loss
            return runner_state, info

        runner_state = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            critic_target_params=critic_target_params,
            actor_target_params=actor_target_params,
            buffer_state=buffer_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
