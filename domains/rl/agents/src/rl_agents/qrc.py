"""Q-learning with Regularized Corrections (QRC) for discrete-action control.

Ghiassian et al., "Gradient Temporal-Difference Learning with Regularized
Corrections", ICML 2020 (arXiv:2007.00611). QRC is the nonlinear control
variant of TDRC: an auxiliary h-head corrects the semi-gradient bias that can
make DQN-style bootstrapping diverge.

Two design decisions carried over from the reference implementation:

- The h-head reads stop-gradiented trunk features (``addHead(..., grad=False)``
  in the reference code) — it is a linear probe on frozen features and never
  shapes the trunk representation.
- There is no target network. QRC is pure gradient TD: the bootstrap uses the
  online parameters, and the ``gamma * sg(h) * v_next`` term in the loss
  supplies the gradient correction a target network would otherwise stand in
  for.

That second decision is why :class:`QRCAgent` cannot be a
:class:`rl_agents.replay_q_agent.ReplayQAgent`, despite being a discrete-action
replay agent with an epsilon schedule: it has no ``TAU``, no target-network
frequency and no ``target_params``, and its loss needs the current ``epsilon``,
which no target rule in that family takes. It also has no eligibility trace to
reset at an episode boundary -- the ``h``-head is an ordinary ``nn.Dense``
living in ``TrainState.params`` and is persistent by design.

:class:`QRCAgent` is the only training path here. The ``make_train`` closure that ran
beside it through the port is gone, and so is ``QRCConfig.GAMMA``: no discount is
configurable on this agent any more. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in replay that
:func:`qrc_loss_batch` reads straight through as its per-transition ``gamma``;
:class:`QRCAgent` never reads ``config.GAMMA``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Literal

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from ml_components.regularization import l2_regularizer
from research_runner.config import traced
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.exploration import linear_epsilon_schedule
from rl_components.heads import epsilon_greedy_action
from rl_components.losses import qrc_terms
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

type QRCApplyFn = Callable[..., tuple[jax.Array, jax.Array]]
"""``QRCNetwork.apply``, as reached through ``TrainState.apply_fn``."""

type HRegularization = Literal["adamw", "l2"]
"""How the h-head's ``BETA`` regularises: decoupled AdamW weight decay (matches the
reference) or an L2 term added directly to the loss (this project's prior behaviour,
which a prior run reported outperforming AdamW for QRC's h-head)."""


@chex_struct(frozen=True, kw_only=True)
class QRCConfig:
    LEARNING_RATE: float = traced(3e-4)
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = traced(1_000)
    TRAIN_FREQUENCY: int = traced(1)
    EPSILON_START: float = traced(1.0)
    EPSILON_END: float = traced(0.05)
    EPSILON_FRACTION: float = traced(0.5)
    BETA: float = traced(1.0)
    H_REGULARIZATION: HRegularization = "adamw"
    HIDDEN_WIDTH: int = 64
    ENV_NAME: str = "CartPole-v1"
    SEED: int = 42

    def build(self) -> QRCAgent:
        return QRCAgent(self)


@chex_struct(frozen=True)
class QRCHypers:
    """The hyperparameters :class:`QRCAgent`'s ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``vmap`` one compiled kernel across a batch of arms. ``TOTAL_TIMESTEPS`` is
    here and not on ``QRCConfig`` alone because the Agent path never scans -- it
    only appears as a divisor in the epsilon schedule, which makes it DYNAMIC
    for this path, matching ``dqn.py``'s classification of the same field once
    it left the scan-owning closure style.
    """

    LEARNING_RATE: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    EPSILON_START: jax.Array
    EPSILON_END: jax.Array
    EPSILON_FRACTION: jax.Array
    TOTAL_TIMESTEPS: jax.Array
    BETA: jax.Array


def qrc_hypers(config: QRCConfig) -> QRCHypers:
    return QRCHypers(
        LEARNING_RATE=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.LEARNING_STARTS, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.TRAIN_FREQUENCY, jnp.int32),
        EPSILON_START=jnp.asarray(config.EPSILON_START, jnp.float32),
        EPSILON_END=jnp.asarray(config.EPSILON_END, jnp.float32),
        EPSILON_FRACTION=jnp.asarray(config.EPSILON_FRACTION, jnp.float32),
        TOTAL_TIMESTEPS=jnp.asarray(config.TOTAL_TIMESTEPS, jnp.float32),
        BETA=jnp.asarray(config.BETA, jnp.float32),
    )


class QRCNetwork(TypedApply[tuple[jax.Array, jax.Array]], nn.Module):
    """Shared trunk with zero-initialised linear q- and h-heads.

    The h-head reads ``jax.lax.stop_gradient``-ed trunk features, so only the
    q-head's loss shapes the trunk; the h-head is a linear probe on frozen
    features.
    """

    action_dim: int
    hidden_width: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        phi = nn.Dense(self.hidden_width)(x)
        phi = nn.relu(phi)
        phi = nn.Dense(self.hidden_width)(phi)
        phi = nn.relu(phi)
        q = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="q_head",
        )(phi)
        h = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            use_bias=False,
            name="h_head",
        )(jax.lax.stop_gradient(phi))
        return q, h


def _param_labels(params: VariableDict) -> VariableDict:
    """"h" for every ``h_head`` leaf, "q" for everything else.

    Selects each parameter's optimizer transform in ``_make_tx``'s
    ``optax.multi_transform``: the h-head gets decoupled weight decay
    (``optax.adamw``), the trunk and q-head do not (Deviation 1). Because
    the h-head reads ``stop_gradient``-ed trunk features and the q-head's
    loss never touches ``h_head`` (Deviation 2 -- see
    ``test_h_head_gradient_does_not_reach_trunk`` and its converse), a
    single gradient of the combined loss already carries, at each leaf,
    exactly that leaf's own term; partitioning the OPTIMIZER is what gives
    the h-head its own (decoupled) update rule without splitting the
    single shared-trunk parameter tree into two.
    """
    def constant_label(_: jax.Array, label: str) -> str:
        return label

    labeled = {
        key: jax.tree.map(partial(constant_label, label="h" if key == "h_head" else "q"), subtree)
        for key, subtree in params["params"].items()
    }
    return {**params, "params": labeled}


def _make_tx(
    params: VariableDict,
    learning_rate: jax.Array | float,
    weight_decay: jax.Array | float,
    h_regularization: HRegularization = "adamw",
) -> optax.GradientTransformation:
    """Adam for the trunk/q-head. AdamW (decoupled weight decay) for the h-head when
    ``h_regularization`` is ``"adamw"``; plain Adam when it is ``"l2"``, since that style
    regularises through an L2 term added to the loss instead (``qrc_loss_batch``).
    """
    h_tx = optax.adamw(learning_rate, weight_decay=weight_decay) if h_regularization == "adamw" else optax.adam(learning_rate)
    return optax.multi_transform({"q": optax.adam(learning_rate), "h": h_tx}, _param_labels(params))


def expected_action_value(q_next: jax.Array, epsilon: jax.Array | float) -> jax.Array:
    """Epsilon-greedy expected next-state value, uniform over the argmax set.

    The greedy distribution is stop-gradiented, so the bootstrap contributes
    gradient only through the action values themselves.
    """
    n_actions = q_next.shape[-1]
    greedy = (q_next == q_next.max()).astype(q_next.dtype)
    pi = greedy / greedy.sum()
    pi = (1.0 - epsilon) * pi + epsilon / n_actions
    pi = jax.lax.stop_gradient(pi)
    return q_next.dot(pi)


def qrc_loss(
    q: jax.Array,
    h: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    gamma: jax.Array,
    q_next: jax.Array,
    epsilon: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-transition QRC loss (Ghiassian et al. 2020, eqs. 8-9).

    ``gamma * sg(delta_hat) * v_next`` is the gradient-TD correction term:
    differentiating it w.r.t. the online parameters yields
    ``gamma * delta_hat * grad(v_next)``, which is exactly the term that
    distinguishes gradient TD from semi-gradient TD and removes the bias
    responsible for divergence under off-policy bootstrapping (e.g. DQN).
    """
    v_next = expected_action_value(q_next, epsilon)
    return qrc_terms(q[action], h[action], reward, gamma, v_next)


def qrc_loss_batch(
    params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    epsilon: jax.Array | float,
    *,
    apply_fn: QRCApplyFn,
    h_regularization: HRegularization = "adamw",
    beta: jax.Array | float = 0.0,
) -> jax.Array:
    """Mean QRC loss over a minibatch.

    Regularising the h-head (Ghiassian et al. 2020, §3.2) is normally the
    caller's optimizer's job, not this loss's: with ``h_regularization="adamw"``,
    ``h_head`` params get decoupled weight decay via their own ``optax.adamw``
    (see ``_make_tx``/``BETA``), not an L2 term added in here -- an in-loss L2
    term would have its gradient rescaled per-parameter by Adam's
    ``1/sqrt(v_hat)``, which is not the same regulariser as AdamW's decoupled
    decay. ``h_regularization="l2"`` adds that L2 term directly instead, using
    ``beta`` as its coefficient, and pairs with plain Adam on the h-head.

    ``discounts`` is the per-transition bootstrap coefficient the loop computed
    and is passed straight through as each transition's ``gamma``, so a terminal
    row's ``0.0`` removes the bootstrap and the correction term with it.

    The network arrives as ``apply_fn`` rather than as a module, because the
    agent's ``step`` reaches it through ``TrainState.apply_fn`` -- a static field
    -- and cannot construct a module from an action count it only carries as a
    traced leaf.
    """
    q, h = apply_fn(params, obs)
    q_next, _ = apply_fn(params, next_obs)

    v_loss, h_loss, _ = jax.vmap(qrc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
        q, h, actions, rewards, discounts, q_next, epsilon
    )

    loss = jnp.mean(v_loss) + jnp.mean(h_loss)
    if h_regularization == "l2":
        loss = loss + beta * l2_regularizer(params, "h_head")
    return loss


@chex_struct(frozen=True)
class QRCAlgorithmState:
    """QRC learner state and swept hyperparameters."""

    train_state: TrainState
    num_actions: jax.Array
    hypers: QRCHypers


class QRCAgent:
    """Q-learning with Regularized Corrections over a discrete action space."""

    REPORTED_METRICS = ("loss", "epsilon", "q_max")
    INIT_SWEPT_FIELDS: tuple[str, ...] = ()
    """Config fields ``init`` reads that ``step`` never traces. None here."""

    def __init__(self, config: QRCConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> ReplayAgentState[QRCAlgorithmState, ReplayBufferState]:
        """Build parameters, optimizer, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description, and the only place the observation shape
                comes from. Must be discrete: the bootstrap is an expectation over a
                finite action set.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space.
        """
        if spec.num_actions is None:
            raise ValueError(f"QRC requires a discrete action space, got spec {spec.id!r} with none")

        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = QRCNetwork(spec.num_actions, hidden_width=self.config.HIDDEN_WIDTH)

        params_key, carry_key = jax.random.split(key)
        buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=tuple(spec.action_shape),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        params = network.init(params_key, jnp.zeros(observation_shape, dtype=observation_dtype))
        h_regularization = self.config.H_REGULARIZATION

        def make_tx(learning_rate: jax.Array | float, weight_decay: jax.Array | float) -> optax.GradientTransformation:
            return _make_tx(params, learning_rate, weight_decay, h_regularization)

        tx = optax.inject_hyperparams(make_tx)(
            learning_rate=jnp.asarray(self.config.LEARNING_RATE, jnp.float32),
            weight_decay=jnp.asarray(self.config.BETA, jnp.float32),
        )

        return init_replay_agent_state(
            QRCAlgorithmState(
                train_state=TrainState.create(apply_fn=network.apply, params=params, tx=tx),
                num_actions=jnp.asarray(spec.num_actions, dtype=jnp.int32),
                hypers=qrc_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: ReplayAgentState[QRCAlgorithmState, ReplayBufferState],
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[ReplayAgentState[QRCAlgorithmState, ReplayBufferState], jax.Array]:
        """Close the pending transition, learn from replay, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes. There is
        no target-network synchronisation step, and nothing to reset at a boundary.

        One epsilon serves both roles it has in this algorithm, as it must: the
        exploration rate and the policy the bootstrap takes its expectation under are the
        same number, and computing them separately is how the two drift apart.

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
        hypers = algorithm_state.hypers
        buffer = ReplayBuffer.from_state(state.replay_state)
        carry_key, sample_key, action_key = jax.random.split(state.key, 3)

        epsilon = linear_epsilon_schedule(
            step_index,
            hypers.TOTAL_TIMESTEPS,
            hypers.EPSILON_START,
            hypers.EPSILON_END,
            hypers.EPSILON_FRACTION,
        )

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

        def select_action(
            learner_state: QRCAlgorithmState,
            observation: jax.Array,
            _current_step: jax.Array,
            keys: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            q_values, _ = learner_state.train_state.apply_fn(learner_state.train_state.params, observation)
            action = epsilon_greedy_action(q_values, epsilon, key=keys[0]).astype(state.last_action.dtype)
            return action, {"epsilon": epsilon, "q_max": jnp.max(q_values).astype(jnp.float32)}

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
            learn=partial(self._learn, buffer=buffer, epsilon=epsilon),
            select_action=select_action,
        )

    def _learn(
        self,
        algorithm_state: QRCAlgorithmState,
        replay_state: ReplayBufferState,
        keys: tuple[jax.Array, ...],
        *,
        buffer: ReplayBuffer,
        epsilon: jax.Array,
    ) -> tuple[QRCAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
        """Take one gradient step on a replay minibatch."""
        hypers = algorithm_state.hypers
        train_state = with_hyperparams(
            algorithm_state.train_state,
            learning_rate=hypers.LEARNING_RATE,
            weight_decay=hypers.BETA,
        )
        obs, actions, rewards, next_obs, discounts = buffer.sample(replay_state, keys[0], self.config.BATCH_SIZE)

        def loss_fn(params: VariableDict) -> jax.Array:
            return qrc_loss_batch(
                params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                epsilon,
                apply_fn=train_state.apply_fn,
                h_regularization=self.config.H_REGULARIZATION,
                beta=hypers.BETA,
            )

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        algorithm_state = replace(algorithm_state, train_state=train_state.apply_gradients(grads=grads))
        return algorithm_state, replay_state, {"loss": loss.astype(jnp.float32)}
