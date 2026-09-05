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

from typing import cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.continuous_actions import continuous_action_dim, uniform_exploration_action
from rl_components.env_protocol import EnvSpec
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

from rl_agents.sac import Actor, sac_alpha_loss


@chex_struct(frozen=True, kw_only=True)
class SACRCConfig:
    LEARNING_RATE: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 5_000
    TRAIN_FREQUENCY: int = 1
    ALPHA: float = 0.2
    TARGET_ENTROPY: float | None = None
    BETA: float = 1.0
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


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
    target = jax.lax.stop_gradient(reward + discount * (next_q_min - alpha * next_log_prob))
    delta = target - q
    delta_hat = h

    v_loss = 0.5 * delta**2 + discount * jax.lax.stop_gradient(delta_hat) * (next_q_min - alpha * next_log_prob)
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return v_loss, h_loss, delta


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

    next_q_values = jax.vmap(
        lambda params, o, a: _critic_apply(critic, params, o, a)[0], in_axes=(0, None, None)
    )(critic_params, next_obs, next_actions)
    next_q_min = jnp.min(next_q_values, axis=0)

    def _single_critic_loss(params: VariableDict) -> jax.Array:
        q, h = _critic_apply(critic, params, obs, actions)
        v_loss, h_loss, _ = jax.vmap(sac_rc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
            q, h, rewards, discounts, next_q_min, next_log_probs, alpha
        )
        h_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params["params"]["h_head"]))
        return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * h_reg

    return jnp.mean(jax.vmap(_single_critic_loss)(critic_params))


def _with_learning_rate(train_state: TrainState, learning_rate: jax.Array) -> TrainState:
    """Push a traced learning rate into optimizer state.

    The transformation itself (``TrainState.tx``) is a static field and cannot hold a
    traced value, so the rate rides ``opt_state.hyperparams`` instead, refreshed each step.
    """
    opt_state = train_state.opt_state
    return train_state.replace(
        opt_state=opt_state._replace(hyperparams={**opt_state.hyperparams, "learning_rate": learning_rate}),
    )


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

    q_values = jax.vmap(
        lambda params, o, a: _critic_apply(critic, params, o, a)[0], in_axes=(0, None, None)
    )(critic_params, obs, new_actions)
    return jnp.mean(alpha * log_probs - jnp.min(q_values, axis=0))


@chex_struct(frozen=True)
class SACRCAgentState:
    """Everything :class:`SACRCAgent` carries between loop iterations.

    There is no ``critic_target_params`` field, and its absence is the algorithm: SAC-RC
    is pure gradient TD, so every bootstrap reads the online critic and the ``h``-head's
    correction term stands in for what a target network would otherwise damp.

    Attributes:
        actor_state: Policy parameters, optimizer state and the static ``apply_fn``.
        critic_state: Twin-critic ensemble parameters, leading axis over members.
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
    alpha_state: TrainState
    buffer_state: ReplayBufferState
    last_obs: jax.Array
    last_action: jax.Array
    target_entropy: jax.Array
    key: chex.PRNGKey
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

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> SACRCAgentState:
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
        return SACRCAgentState(
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
            buffer_state=buffer.init(),
            last_obs=zero_obs,
            last_action=zero_action,
            target_entropy=jnp.asarray(target_entropy, dtype=jnp.float32),
            key=carry_key,
            hypers=sac_rc_hypers(self.config),
        )

    def step(
        self,
        state: SACRCAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[SACRCAgentState, jax.Array]:
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
        hypers = state.hypers
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

        can_train = (step_index > hypers.LEARNING_STARTS) & (step_index % hypers.TRAIN_FREQUENCY == 0)
        (
            actor_state,
            critic_state,
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
            state=SACRCAgentState(
                actor_state=actor_state,
                critic_state=critic_state,
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
        state: SACRCAgentState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
    ) -> tuple[TrainState, TrainState, TrainState, jax.Array, jax.Array, jax.Array]:
        """Take one critic, actor and temperature gradient step on a replay minibatch.

        The rate is pushed into each optimizer's state each step because the
        transformation itself is a static field and cannot hold a traced value.
        """
        hypers = state.hypers
        actor, critic = _networks(state.last_action.shape[0])
        sample_key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, sample_key, self.config.BATCH_SIZE)
        alpha = jnp.exp(cast(jax.Array, state.alpha_state.params["log_alpha"])[0])

        critic_state = _with_learning_rate(state.critic_state, hypers.LEARNING_RATE)
        actor_state = _with_learning_rate(state.actor_state, hypers.LEARNING_RATE)
        alpha_state = _with_learning_rate(state.alpha_state, hypers.LEARNING_RATE)

        critic_loss, critic_grads = jax.value_and_grad(
            lambda params: sac_rc_loss_batch(
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
        )(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        actor_loss, actor_grads = jax.value_and_grad(
            lambda params: sac_rc_actor_loss(
                params, critic_state.params, obs, alpha, actor_key, actor=actor, critic=critic
            )
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        alpha_loss, alpha_grads = jax.value_and_grad(
            lambda params: sac_alpha_loss(
                params, actor_state.params, obs, state.target_entropy, alpha_key, actor=actor
            )
        )(alpha_state.params)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return (
            actor_state,
            critic_state,
            alpha_state,
            critic_loss.astype(jnp.float32),
            actor_loss.astype(jnp.float32),
            alpha_loss.astype(jnp.float32),
        )

