"""Soft Actor-Critic with Regularized Corrections (SAC-RC) for continuous-action control.

The continuous-action analogue of ``rl_agents.qrc``: SAC's twin-critic
ensemble gets the same TDRC/QRC gradient-TD correction QRC applies to DQN
(Ghiassian et al., "Gradient Temporal-Difference Learning with Regularized
Corrections", ICML 2020, arXiv:2007.00611).

Two design decisions carried over from ``qrc.py``:

- The h-head reads stop-gradiented trunk features — it is a linear probe on
  frozen features and never shapes the trunk/q-head representation.
- There is no target network. SAC-RC is pure gradient TD: every bootstrap
  uses the online critic parameters, and the ``gamma * sg(h) * bootstrap``
  term in the loss supplies the gradient correction a target network would
  otherwise stand in for. Stop-gradienting that bootstrap would silently
  collapse SAC-RC back into plain semi-gradient SAC with an inert extra head.

The squared losses carry a ``0.5`` that TDRC's published form omits. That is
a house convention, but it scales the data terms and not ``BETA``'s
regulariser, so a given ``BETA`` is twice as strong relative to the data as
the same number in the published sweeps.

The entropy-coefficient (alpha) update is not merely equivalent to
``rl_agents.sac``'s, it *is* :func:`rl_agents.sac.sac_alpha_loss`: it reads only
the policy, which SAC-RC does not change. The actor loss needs its own function
because this critic returns ``(q, h)`` rather than ``q``, but it is SAC's
expression over the q-head.

:class:`SACRCAgent` is the only training path here. The ``make_train`` closure
that ran beside it through the port is gone, and so is ``SACRCConfig.GAMMA``:
no discount is configurable on this agent any more. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in
replay that :class:`SACRCAgent` reads.

SAC-RC was built on the ``make_train`` path during this migration and is absent
from the agent-port specification's ten-agent plan, so its port was scope the
spec did not anticipate. It is here because leaving one continuous agent on the
old path would have kept the legacy loop alive after every agent the spec does
name had left it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from ml_components.regularization import l2_regularizer
from research_runner.config import traced
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.continuous_actions import continuous_action_dim
from rl_components.env_protocol import EnvSpec
from rl_components.losses import qrc_terms
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

from rl_agents.sac import Actor, sac_alpha_loss


@chex_struct(frozen=True, kw_only=True)
class SACRCConfig:
    LEARNING_RATE: float = traced(3e-4)
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = traced(5_000)
    TRAIN_FREQUENCY: int = traced(1)
    ALPHA: float = traced(0.2)
    TARGET_ENTROPY: float | None = None
    BETA: float = traced(1.0)
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42

    def build(self) -> SACRCAgent:
        return SACRCAgent(self)


@chex_struct(frozen=True)
class SACRCHypers:
    """The hyperparameters :class:`SACRCAgent`'s ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. ``ALPHA`` is not here:
    the entropy coefficient the loss actually uses comes from the learned
    ``alpha_state.params["log_alpha"]``, never from config -- see the module
    docstring's "not merely equivalent" note and ``sac.py``'s dead-field flag
    on the same name.
    """

    LEARNING_RATE: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    BETA: jax.Array


def sac_rc_hypers(config: SACRCConfig) -> SACRCHypers:
    return SACRCHypers(
        LEARNING_RATE=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.LEARNING_STARTS, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.TRAIN_FREQUENCY, jnp.int32),
        BETA=jnp.asarray(config.BETA, jnp.float32),
    )


class SACRCCritic(nn.Module):
    """Twin-critic trunk (matching ``sac.Critic``) plus a QRC-style h-head.

    The h-head reads ``jax.lax.stop_gradient``-ed trunk features, and is
    zero-initialised and bias-free, exactly as in ``qrc.QRCNetwork`` — it is
    a linear probe on frozen features that starts at zero and never shapes
    the trunk. The trunk and q-head match ``sac.Critic`` (default init).
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        xa = jnp.concatenate([x, a], axis=-1)
        phi = nn.Dense(256)(xa)
        phi = nn.relu(phi)
        phi = nn.Dense(256)(phi)
        phi = nn.relu(phi)
        q = nn.Dense(1, name="q_head")(phi)
        h = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            use_bias=False,
            name="h_head",
        )(jax.lax.stop_gradient(phi))
        return jnp.squeeze(q, axis=-1), jnp.squeeze(h, axis=-1)


def _critic_apply(module: SACRCCritic, variables: VariableDict, x: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return cast(tuple[jax.Array, jax.Array], module.apply(variables, x, a))


def sac_rc_loss(
    q: jax.Array,
    h: jax.Array,
    reward: jax.Array,
    discount: jax.Array,
    next_q_min: jax.Array,
    next_log_prob: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-transition, per-ensemble-member SAC-RC loss.

    ``bootstrap`` is the soft value ``min_j Q_j(s', a') - alpha * log pi(a'|s')``
    computed from the ONLINE critic (no target network). ``target`` stop-
    gradients that bootstrap for the semi-gradient TD term, but the
    correction term ``gamma * sg(h) * bootstrap`` reuses the un-stopped
    ``bootstrap`` so its gradient reaches the online parameters that produced
    it: differentiating it w.r.t. those parameters gives
    ``gamma * delta_hat * grad(bootstrap)``, the term that distinguishes
    gradient TD from semi-gradient TD and removes the divergence-inducing
    off-policy bootstrapping bias that a target network is normally used to
    paper over.
    """
    bootstrap = next_q_min - alpha * next_log_prob
    return qrc_terms(q, h, reward, discount, bootstrap)


def sac_rc_loss_batch(
    critic_params: VariableDict,
    critic: SACRCCritic,
    actor_params: VariableDict,
    actor: Actor,
    alpha: jax.Array,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    rng: jax.Array,
    beta: jax.Array | float,
) -> jax.Array:
    """Mean SAC-RC critic loss over a minibatch, plus L2 regularisation on the h-heads.

    ``next_q_min`` is shared across ensemble members (the min is over the
    ensemble), each member's own ``q``/``h`` are read from its own params via
    ``vmap``, matching the twin-critic structure of ``sac.py``.
    """
    next_actions, next_log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, next_obs, jax.random.split(rng, obs.shape[0])
    )

    def member_q(params: VariableDict, o: jax.Array, a: jax.Array) -> jax.Array:
        return _critic_apply(critic, params, o, a)[0]

    next_q_values = jax.vmap(member_q, in_axes=(0, None, None))(critic_params, next_obs, next_actions)
    next_q_min = jnp.min(next_q_values, axis=0)

    def _single_critic_loss(params: VariableDict) -> jax.Array:
        q, h = _critic_apply(critic, params, obs, actions)
        v_loss, h_loss, _ = jax.vmap(sac_rc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
            q, h, rewards, discounts, next_q_min, next_log_probs, alpha
        )
        return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * l2_regularizer(params, "h_head")

    return jnp.mean(jax.vmap(_single_critic_loss)(critic_params))


def _networks(action_dim: int) -> tuple[Actor, SACRCCritic]:
    """Build the two Flax modules, which carry no state beyond the action dimension.

    ``init`` reads that dimension from the environment spec and ``step`` reads it back off
    the shape of the action it carries, so neither has to store the modules themselves.
    """
    return Actor(action_dim), SACRCCritic()


def sac_rc_actor_loss(
    actor_params: VariableDict,
    critic_params: VariableDict,
    obs: jax.Array,
    alpha: jax.Array,
    key: chex.PRNGKey,
    *,
    actor: Actor,
    critic: SACRCCritic,
) -> jax.Array:
    """SAC's actor loss, read off this critic's q-head.

    The h-head plays no part: it corrects the critic's own bootstrap and is not a value
    the policy should climb.

    Args:
        actor_params: Policy parameters. Differentiate with respect to this argument only.
        critic_params: Online critic ensemble, whose q-head minimum scores the actions.
        obs: Observations to act from.
        alpha: Entropy temperature.
        key: PRNG key for the action sample.
        actor: The policy module, static because it is a Python object.
        critic: The critic module, likewise.

    Returns:
        The scalar loss.
    """
    _, sample_key = jax.random.split(key)
    new_actions, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, obs, jax.random.split(sample_key, obs.shape[0])
    )

    def member_q(params: VariableDict, o: jax.Array, a: jax.Array) -> jax.Array:
        return _critic_apply(critic, params, o, a)[0]

    q_values = jax.vmap(member_q, in_axes=(0, None, None))(critic_params, obs, new_actions)
    return jnp.mean(alpha * log_probs - jnp.min(q_values, axis=0))


@chex_struct(frozen=True)
class SACRCAlgorithmState:
    """SAC-RC learner state carried by the shared replay-agent wrapper."""

    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    target_entropy: jax.Array
    hypers: SACRCHypers


class SACRCAgent:
    """SAC with the QRC gradient correction and no target network.

    The policy is tanh-squashed, so the environment's action space must already be
    normalized to ``[-1, 1]``; see :mod:`rl_components.continuous_actions`.
    """

    REPORTED_METRICS = ("critic_loss", "actor_loss", "alpha_loss", "alpha")
    INIT_SWEPT_FIELDS: tuple[str, ...] = ("ALPHA", "TARGET_ENTROPY")
    """Config fields ``init`` reads but ``step`` never traces, so they ride neither
    ``SACRCHypers`` nor a per-step tracer -- yet ``init`` itself runs inside the vmapped
    region exactly as ``step`` does, so a per-arm traced value works there too, making
    both genuinely sweepable despite never appearing on the hypers struct.

    ``TARGET_ENTROPY`` is ``float | None``: ``None`` means "derive ``-action_dim`` in
    ``init``". A sweep must supply a numeric value for every arm -- it cannot mix ``None``
    with numbers in one vmapped batch -- so sweeping this field forgoes the default-entropy
    behavior for the arms in that sweep. That limitation is not solved here.
    """

    def __init__(self, config: SACRCConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> ReplayAgentState[SACRCAlgorithmState, ReplayBufferState]:
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
        buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=(action_dim,),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )
        target_entropy = -float(action_dim) if self.config.TARGET_ENTROPY is None else self.config.TARGET_ENTROPY

        lr = jnp.asarray(self.config.LEARNING_RATE, jnp.float32)
        return init_replay_agent_state(
            SACRCAlgorithmState(
                actor_state=TrainState.create(
                    apply_fn=actor.apply,
                    params=actor.init(actor_key, zero_obs),
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                critic_state=TrainState.create(
                    apply_fn=critic.apply,
                    params=jax.vmap(critic.init, in_axes=(0, None, None))(
                        jax.random.split(critic_key, 2), zero_obs, zero_action
                    ),
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                alpha_state=TrainState.create(
                    apply_fn=None,
                    params={"log_alpha": jnp.log(jnp.array([self.config.ALPHA], dtype=jnp.float32))},
                    tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr),
                ),
                target_entropy=jnp.asarray(target_entropy, dtype=jnp.float32),
                hypers=sac_rc_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: ReplayAgentState[SACRCAlgorithmState, ReplayBufferState],
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[ReplayAgentState[SACRCAlgorithmState, ReplayBufferState], jax.Array]:
        """Close the pending transition, learn from replay, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes. There is
        no target-network synchronisation step, by design.

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
            learner_state: SACRCAlgorithmState,
            replay_state: ReplayBufferState,
            keys: tuple[jax.Array, ...],
        ) -> tuple[SACRCAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
            return self._learn(learner_state, replay_state, buffer, keys[0], action_dim)

        def select_action(
            learner_state: SACRCAlgorithmState,
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
        learner_state: SACRCAlgorithmState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        action_dim: int,
    ) -> tuple[SACRCAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
        """Take one critic, actor and temperature gradient step on a replay minibatch.

        The rate is pushed into each optimizer's state each step because the
        transformation itself is a static field and cannot hold a traced value.
        """
        hypers = learner_state.hypers
        actor, critic = _networks(action_dim)
        sample_key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        _, alpha_sample_key = jax.random.split(alpha_key)
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, sample_key, self.config.BATCH_SIZE)
        alpha = jnp.exp(cast(jax.Array, learner_state.alpha_state.params["log_alpha"])[0])

        critic_state = with_hyperparams(learner_state.critic_state, learning_rate=hypers.LEARNING_RATE)
        actor_state = with_hyperparams(learner_state.actor_state, learning_rate=hypers.LEARNING_RATE)
        alpha_state = with_hyperparams(learner_state.alpha_state, learning_rate=hypers.LEARNING_RATE)

        def critic_loss_fn(params: VariableDict) -> jax.Array:
            return sac_rc_loss_batch(
                params,
                critic,
                actor_state.params,
                actor,
                alpha,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                critic_key,
                hypers.BETA,
            )

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        def actor_loss_fn(params: VariableDict) -> jax.Array:
            return sac_rc_actor_loss(params, critic_state.params, obs, alpha, actor_key, actor=actor, critic=critic)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        def alpha_loss_fn(params: VariableDict) -> jax.Array:
            return sac_alpha_loss(
                params, actor_state.params, obs, learner_state.target_entropy, alpha_sample_key, actor=actor
            )

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        updated_state = replace(
            learner_state,
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )
        return updated_state, buffer_state, {
            "critic_loss": critic_loss.astype(jnp.float32),
            "actor_loss": actor_loss.astype(jnp.float32),
            "alpha_loss": alpha_loss.astype(jnp.float32),
            "alpha": jnp.exp(cast(jax.Array, alpha_state.params["log_alpha"])[0]).astype(jnp.float32),
        }
