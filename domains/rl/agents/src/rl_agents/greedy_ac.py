"""Greedy Actor-Critic (GAC) — percentile actor + QRC critic for continuous control.

Faithful implementation of the GAC algorithm.

Key aspects:
- Percentile Actor: learns from top-k%% actions ranked by Q-value
- QRC Critic: single critic with auxiliary h-network (TDC-style learning)
- Sampling-based proposal selection (random + uniform mixin)
- Simple MLP architecture
- Continuous actions via TanhNormalDiag (tanh-squashed, [-1, 1])

References:
- GAC: Neumann et al., "Greedy Actor-Critic: A New Conditional Cross-Entropy
  Method for Policy Improvement", ICLR 2023 (arXiv:1810.09103)
- QRC: Ghiassian et al., "Gradient Temporal-Difference Learning with
  Regularized Corrections", ICML 2020 (arXiv:2007.00611). QRC is the
  nonlinear control variant of TDRC; the auxiliary h-network and its L2
  term come from there.

:class:`GACAgent` is the only training path here. The ``make_train`` closure that ran
beside it through the port is gone, and so is ``GACConfig.GAMMA``: no discount is
configurable on this agent any more. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in replay,
and that coefficient is read straight through as each transition's ``gamma`` in
:func:`gac_critic_loss`. Two sources of truth for the discount is how the bootstrap
defect this port exists to fix recurs. Like ``qrc``, GAC has no eligibility trace to
reset at an episode boundary: the ``h``-head is an ordinary ``nn.Dense`` inside the
critic's ``TrainState.params`` and is persistent by design.

Both batch losses take their network as an argument rather than constructing
one. They used to build ``GACCritic()`` and ``GACActor(action_dim)`` with
default widths, which silently disagreed with the trained parameters for any
``HIDDEN_SIZE`` other than 256; the port needs the real network and so passes
it in.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import NamedTuple, cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_utils.pytree import tree_norm
from ml_components.regularization import l2_regularizer
from research_runner.config import traced
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.continuous_actions import continuous_action_dim
from rl_components.distributions import TanhNormalDiag
from rl_components.env_protocol import EnvSpec
from rl_components.losses import qrc_terms
from rl_components.optim import with_hyperparams
from rl_components.replay_agent import ReplayAgentState, init_replay_agent_state, replay_agent_step
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════


@chex_struct(frozen=True, kw_only=True)
class GACConfig:
    """Hyperparameters for the Greedy Actor-Critic algorithm.

    Attributes:
        learning_rate: Critic learning rate.
        ACTOR_LEARNING_RATE: Actor learning rate.
        BUFFER_SIZE: Capacity of the uniform replay buffer.
        BATCH_SIZE: Minibatch size for both actor and critic updates.
        total_timesteps: Total environment steps per training run.
        LEARNING_STARTS: Steps of random exploration before training begins.
        TRAIN_FREQUENCY: Number of environment steps between training updates.
        TAU: Target network polyak averaging coefficient.
        NUM_SAMPLES: Number of action proposals per state for actor update.
        ACTOR_PERCENTILE: Fraction of top-ranked proposals used for actor training.
        UNIFORM_WEIGHT: Fraction of proposals that are uniform random (vs. actor-sampled).
        ENTROPY_WEIGHT: Bonus weight for action entropy in actor loss.
        NUM_RAND_ACTIONS: Number of action samples for next-state value estimation.
        TRAINING_SIGMA_MIN: Minimum std dev for the actor during training.
        INFERENCE_SIGMA_MIN: Minimum std dev for the actor during inference.
        H_REGULARIZATION: L2 regularization weight for h-network parameters
            (QRC paper default 1.0 — only applied to the tiny h-head, ~256 params).
        HIDDEN_SIZE: Width of hidden layers in actor and critic networks.
        ENV_NAME: Gymnax environment name.
        SEED: Random seed for reproducibility.
    """

    learning_rate: float = traced(3e-4)
    ACTOR_LEARNING_RATE: float = traced(3e-4)
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    total_timesteps: int = 200_000
    LEARNING_STARTS: int = traced(1_000)
    TRAIN_FREQUENCY: int = traced(1)
    TAU: float = 0.005
    NUM_SAMPLES: int = 32
    ACTOR_PERCENTILE: float = 0.1
    UNIFORM_WEIGHT: float = 0.2
    ENTROPY_WEIGHT: float = traced(0.01)
    NUM_RAND_ACTIONS: int = 10
    TRAINING_SIGMA_MIN: float = traced(5e-3)
    INFERENCE_SIGMA_MIN: float = traced(1e-4)
    H_REGULARIZATION: float = traced(1.0)
    HIDDEN_SIZE: int = 256
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42

    def build(self) -> GACAgent:
        return GACAgent(self)


# ═══════════════════════════════════════════════════════════════
# Critic Network — QRC (simple MLP + Q-head + h-head)
# ═══════════════════════════════════════════════════════════════


class GACCritic(nn.Module):
    """QRC critic — 2-layer MLP with Q-head and named h-head.

    Concatenates state + action, passes through 2 hidden layers,
    then splits into Q-value and h-value heads.
    The ``h_head`` submodule is named so its params can be extracted
    for L2 regularization.
    """

    hidden_size: int = 256

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        q = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01), use_bias=False)(x)
        h = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01), use_bias=False, name="h_head")(x)
        return jnp.squeeze(q, axis=-1), jnp.squeeze(h, axis=-1)


# ═══════════════════════════════════════════════════════════════
# Actor Network — Gaussian policy (simple MLP)
# ═══════════════════════════════════════════════════════════════


class GACActor(nn.Module):
    """Gaussian policy — 2-layer MLP outputting mean + log_std.

    Outputs a ``TanhNormalDiag`` distribution over actions in [-1, 1].
    """

    action_dim: int
    hidden_size: int = 256

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mu, log_std


# ── Apply helpers ────────────────────────────────────────────


def _critic_apply(module: GACCritic, params: VariableDict, x: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return cast(tuple[jax.Array, jax.Array], module.apply(params, x, a))


def _actor_apply(module: GACActor, params: VariableDict, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return cast(tuple[jax.Array, jax.Array], module.apply(params, x))


def _make_dist(mean: jax.Array, log_std: jax.Array, sigma_min: float | jax.Array) -> TanhNormalDiag:
    """Build a TanhNormalDiag from actor outputs, adding sigma_min floor."""
    log_std = jnp.clip(log_std, -20.0, 2.0)
    std = jnp.exp(log_std) + sigma_min
    log_std = jnp.log(std)
    return TanhNormalDiag(mean=mean, log_std=log_std, epsilon=1e-6)


def actor_sample(
    actor: GACActor,
    params: VariableDict,
    x: jax.Array,
    rng: jax.Array,
    sigma_min: float | jax.Array,
    n: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Sample ``n`` actions from the actor's policy (via vmap over PRNG splits)."""
    mean, log_std = _actor_apply(actor, params, x)
    dist = _make_dist(mean, log_std, sigma_min)
    keys = jax.random.split(rng, n)

    def sample_action(k: jax.Array) -> jax.Array:
        return dist.sample(seed=k)

    actions = jax.vmap(sample_action)(keys)
    log_probs = jax.vmap(dist.log_prob)(actions)
    return actions, log_probs


def actor_mean_log_prob(
    actor: GACActor,
    params: VariableDict,
    x: jax.Array,
    actions: jax.Array,
    sigma_min: float | jax.Array,
) -> jax.Array:
    """Compute log-probability of given actions under the actor's policy."""
    mean, log_std = _actor_apply(actor, params, x)
    dist = _make_dist(mean, log_std, sigma_min)
    return dist.log_prob(actions)


def actor_entropy(
    actor: GACActor,
    params: VariableDict,
    x: jax.Array,
    sigma_min: float | jax.Array,
) -> jax.Array:
    """Compute the entropy of the actor's policy."""
    mean, log_std = _actor_apply(actor, params, x)
    dist = _make_dist(mean, log_std, sigma_min)
    return dist.entropy()


# ═══════════════════════════════════════════════════════════════
# State types
# ═══════════════════════════════════════════════════════════════


class QRCTransition(NamedTuple):
    """Single transition for the QRC critic loss."""
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    gamma: jax.Array


class QRCMetrics(NamedTuple):
    """Per-element critic loss breakdown."""
    q: jax.Array
    h: jax.Array
    loss: jax.Array
    q_loss: jax.Array
    h_loss: jax.Array
    delta: jax.Array


# ═══════════════════════════════════════════════════════════════
# QRC Critic Loss (element-wise, batched via vmap)
# ═══════════════════════════════════════════════════════════════


def _qrc_loss(
    critic_params: VariableDict,
    transition: QRCTransition,
    next_action: jax.Array,
    *,
    critic: GACCritic,
) -> tuple[jax.Array, QRCMetrics]:
    r"""QRC loss for a single transition (TDC-style).

    Loss structure:

    .. math::

        \delta_l &= \operatorname{sg}(q_{\text{target}}) - q \\
        \delta_r &= q_{\text{target}} - \operatorname{sg}(q) \\
        \mathcal{L}_q &= \frac{1}{2}\delta_l^2 + \operatorname{sg}(h) \cdot \delta_r \\
        \mathcal{L}_h &= \frac{1}{2}(\operatorname{sg}(\delta_l) - h)^2 \\
        \mathcal{L} &= \mathcal{L}_q + \mathcal{L}_h

    where :math:`q_{\text{target}} = r + \gamma q(s', a')`.

    The ``h``-network provides a gradient correction analogous to
    Gradient-TD / TDC, reducing bias from the stopped target.
    """
    obs, action, reward, next_obs, gamma = transition
    action = action.reshape((-1,))
    next_action = next_action.reshape((-1,))

    q, h = _critic_apply(critic, critic_params, obs, action)
    q_prime, _ = _critic_apply(critic, critic_params, next_obs, next_action)

    q_loss, h_loss, delta_l = qrc_terms(q, h, reward, gamma, q_prime)
    loss = q_loss + h_loss

    metrics = QRCMetrics(
        q=q,
        h=h,
        loss=loss,
        q_loss=q_loss,
        h_loss=h_loss,
        delta=delta_l,
    )
    return loss, metrics


def gac_critic_loss(
    critic_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    next_actions: jax.Array,
    reg_weight: float | jax.Array,
    *,
    critic: GACCritic,
) -> tuple[jax.Array, QRCMetrics]:
    """Vectorised QRC loss over a batch of transitions.

    L2 regularization is applied **only** to the h-head parameters
    (QRC paper §3.2).

    ``discounts`` is the per-transition bootstrap coefficient the loop
    computed, passed straight through as each transition's ``gamma``, so a
    terminal row's ``0.0`` removes the bootstrap and the correction with it.
    """
    transitions = jax.vmap(QRCTransition)(obs, actions, rewards, next_obs, discounts)

    losses, metrics = jax.vmap(partial(_qrc_loss, critic=critic), in_axes=(None, 0, 0))(
        critic_params, transitions, next_actions,
    )

    # L2 regularisation on h-head only (QRC paper §3.2)
    h_reg = l2_regularizer(critic_params, "h_head")

    return jnp.mean(losses) + reg_weight * h_reg, jax.tree.map(jnp.mean, metrics)


# ═══════════════════════════════════════════════════════════════
# Sampling-based proposal helpers
# ═══════════════════════════════════════════════════════════════


def _propose_and_rank_topk(
    critic_params: VariableDict,
    actor_params: VariableDict,
    obs: jax.Array,            # (B, obs_dim)
    rng: jax.Array,            # (B, 2) — one PRNG per batch element
    num_samples: int,
    uniform_weight: float,
    actor_percentile: float,
    action_dim: int,
    sigma_min: float | jax.Array,
    *,
    critic: GACCritic,
    actor: GACActor,
) -> jax.Array:
    """Sampling-based percentile action selection.

    For each of ``B`` states in parallel:
    1. Generate ``num_samples`` proposals (uniform + actor-sampled)
    2. Evaluate all ``(B × num_samples)`` proposals in one flat vmap
    3. Pick top-k per state

    Returns:
        Top-k actions, shape ``(B, k, action_dim)``.
    """
    B = obs.shape[0]
    # STATIC BY NECESSITY: n_unif sizes jax.random.uniform's array and the concatenate
    # split point below, both of which must be Python ints at trace time. uniform_weight
    # therefore cannot ride the hypers pytree without a redesign (e.g. a soft mask over a
    # fixed NUM_SAMPLES) -- out of scope here, so it stays on GACConfig.
    n_unif = max(1, int(num_samples * uniform_weight))
    n_prop = num_samples - n_unif

    def _proposals_for_one(state: jax.Array, key: jax.Array) -> jax.Array:
        unif_key, prop_key = jax.random.split(key)
        unif = jax.random.uniform(unif_key, (n_unif, action_dim), minval=-1.0, maxval=1.0)
        if n_prop > 0:
            prop, _ = actor_sample(actor, actor_params, state, prop_key, sigma_min, n=n_prop)
            return jnp.concatenate([unif, prop], axis=0)
        return unif

    proposals = jax.vmap(_proposals_for_one)(obs, rng)  # (B, N, D)

    # Flat vmap: evaluate all (B × N) proposals in one pass
    total = B * num_samples
    flat_proposals = proposals.reshape(total, action_dim)
    flat_obs = jnp.repeat(obs, num_samples, axis=0)

    def critic_q(s: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _critic_apply(critic, critic_params, s, a)

    flat_q, _ = jax.vmap(critic_q)(flat_obs, flat_proposals)

    q_values = flat_q.reshape(B, num_samples)
    # STATIC BY NECESSITY: k is jax.lax.top_k's shape-determining argument, so it must be a
    # Python int at trace time. actor_percentile therefore cannot ride the hypers pytree
    # without a redesign (e.g. a soft top-k mask over a fixed NUM_SAMPLES) -- out of scope
    # here, so it stays on GACConfig.
    k = max(1, min(int(actor_percentile * num_samples), num_samples))
    _, top_idxs = jax.lax.top_k(q_values, k)

    def gather_top(p: jax.Array, idx: jax.Array) -> jax.Array:
        return p[idx]

    return jax.vmap(gather_top)(proposals, top_idxs)


def _select_best_next_action(
    critic_params: VariableDict,
    actor_params: VariableDict,
    next_obs: jax.Array,       # (B, obs_dim)
    rngs: jax.Array,           # (B, 2) — one PRNG per batch element
    num_rand_actions: int,
    action_dim: int,
    sigma_min: float | jax.Array,
    *,
    critic: GACCritic,
    actor: GACActor,
) -> jax.Array:
    """For each next-state, sample N actions and pick the one with highest Q.

    Flat vmap over all ``(B × N)`` proposals.
    Returns shape ``(B, action_dim)``.
    """
    B = next_obs.shape[0]
    N = num_rand_actions

    def _sample_one(state: jax.Array, key: jax.Array) -> jax.Array:
        return actor_sample(actor, actor_params, state, key, sigma_min, n=N)[0]

    proposals = jax.vmap(_sample_one)(next_obs, rngs)   # (B, N, D)

    total = B * N
    flat_proposals = proposals.reshape(total, action_dim)
    flat_obs = jnp.repeat(next_obs, N, axis=0)

    def critic_q(s: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _critic_apply(critic, critic_params, s, a)

    flat_q, _ = jax.vmap(critic_q)(flat_obs, flat_proposals)

    q_values = flat_q.reshape(B, N)
    best_idxs = jnp.argmax(q_values, axis=-1)

    def gather_best(p: jax.Array, idx: jax.Array) -> jax.Array:
        return p[idx]

    return jax.vmap(gather_best)(proposals, best_idxs)


# ═══════════════════════════════════════════════════════════════
# Actor Loss
# ═══════════════════════════════════════════════════════════════


def _actor_loss(
    actor_params: VariableDict,
    state_features: jax.Array,
    top_actions: jax.Array,
    entropy_weight: float | jax.Array,
    sigma_min: float | jax.Array,
    *,
    actor: GACActor,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Negative log-likelihood on top actions + entropy bonus."""
    log_prob = actor_mean_log_prob(actor, actor_params, state_features, top_actions, sigma_min)
    nll = -jnp.mean(log_prob)
    ent = jnp.sum(actor_entropy(actor, actor_params, state_features, sigma_min))
    loss = nll - entropy_weight * ent
    return loss, {"nll": nll, "entropy": ent}


def gac_actor_loss(
    actor_params: VariableDict,
    state_features: jax.Array,
    top_actions_batch: jax.Array,  # (batch, k, action_dim)
    entropy_weight: float | jax.Array,
    sigma_min: float | jax.Array,
    *,
    actor: GACActor,
) -> jax.Array:
    """Mean actor loss across a batch of states."""

    def per_state_loss(sf: jax.Array, ta: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _actor_loss(actor_params, sf, ta, entropy_weight, sigma_min, actor=actor)

    losses, _ = jax.vmap(per_state_loss)(state_features, top_actions_batch)
    return jnp.mean(losses)


# ═══════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════


@chex_struct(frozen=True)
class GACHypers:
    """The hyperparameters ``GACAgent.step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can ``vmap`` one
    compiled kernel across a batch of arms. ``ACTOR_PERCENTILE`` and ``UNIFORM_WEIGHT`` are
    deliberately absent: both are cast through Python ``int()`` to size an array (see the
    call sites in ``_propose_and_rank_topk``), so sweeping them would need a redesign and
    stays out of scope here. ``TOTAL_TIMESTEPS`` is absent too: :class:`GACAgent` never
    reads it.
    """

    LEARNING_RATE: jax.Array
    ACTOR_LEARNING_RATE: jax.Array
    LEARNING_STARTS: jax.Array
    TRAIN_FREQUENCY: jax.Array
    ENTROPY_WEIGHT: jax.Array
    TRAINING_SIGMA_MIN: jax.Array
    INFERENCE_SIGMA_MIN: jax.Array
    H_REGULARIZATION: jax.Array


def gac_hypers(config: GACConfig) -> GACHypers:
    return GACHypers(
        LEARNING_RATE=jnp.asarray(config.learning_rate, jnp.float32),
        ACTOR_LEARNING_RATE=jnp.asarray(config.ACTOR_LEARNING_RATE, jnp.float32),
        LEARNING_STARTS=jnp.asarray(config.LEARNING_STARTS, jnp.int32),
        TRAIN_FREQUENCY=jnp.asarray(config.TRAIN_FREQUENCY, jnp.int32),
        ENTROPY_WEIGHT=jnp.asarray(config.ENTROPY_WEIGHT, jnp.float32),
        TRAINING_SIGMA_MIN=jnp.asarray(config.TRAINING_SIGMA_MIN, jnp.float32),
        INFERENCE_SIGMA_MIN=jnp.asarray(config.INFERENCE_SIGMA_MIN, jnp.float32),
        H_REGULARIZATION=jnp.asarray(config.H_REGULARIZATION, jnp.float32),
    )


def _networks(action_dim: int, hidden_size: int) -> tuple[GACActor, GACCritic]:
    """Build the two Flax modules from the two static facts that shape them.

    ``init`` reads the action dimension from the environment spec and ``step`` reads it
    back off the shape of the action it carries; the width is a configuration constant.
    Neither the width nor the dimension is a traced value, so neither has to be stored.
    """
    return GACActor(action_dim, hidden_size=hidden_size), GACCritic(hidden_size=hidden_size)


@chex_struct(frozen=True)
class GACAlgorithmState:
    """GAC learner state carried by the shared replay-agent wrapper."""

    actor_state: TrainState
    critic_state: TrainState
    hypers: GACHypers


class GACAgent:
    """Greedy Actor-Critic: a percentile actor trained on the critic's own top proposals.

    The policy is tanh-squashed, so the environment's action space must already be
    normalized to ``[-1, 1]``; see :mod:`rl_components.continuous_actions`.
    """

    REPORTED_METRICS = ("critic_loss", "actor_loss", "critic_grad_norm", "actor_grad_norm")
    INIT_SWEPT_FIELDS: tuple[str, ...] = ()
    """Config fields ``init`` reads that ``step`` never traces. None here."""

    def __init__(self, config: GACConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> ReplayAgentState[GACAlgorithmState, ReplayBufferState]:
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
        actor, critic = _networks(action_dim, self.config.HIDDEN_SIZE)

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

        return init_replay_agent_state(
            GACAlgorithmState(
                actor_state=TrainState.create(
                    apply_fn=actor.apply,
                    params=actor.init(actor_key, zero_obs),
                    tx=optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.inject_hyperparams(optax.adamw)(
                            learning_rate=jnp.asarray(self.config.ACTOR_LEARNING_RATE, jnp.float32), weight_decay=0.1
                        ),
                    ),
                ),
                critic_state=TrainState.create(
                    apply_fn=critic.apply,
                    params=critic.init(critic_key, zero_obs, zero_action),
                    tx=optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.inject_hyperparams(optax.adamw)(
                            learning_rate=jnp.asarray(self.config.learning_rate, jnp.float32), weight_decay=0.1
                        ),
                    ),
                ),
                hypers=gac_hypers(self.config),
            ),
            buffer.init(),
            carry_key,
            spec,
        )

    def step(
        self,
        state: ReplayAgentState[GACAlgorithmState, ReplayBufferState],
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[ReplayAgentState[GACAlgorithmState, ReplayBufferState], jax.Array]:
        """Close the pending transition, learn from replay, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes. There is
        no target-network synchronisation step, and nothing to reset at a boundary.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so both
                insertion and learning are guarded above it.

        Returns:
            The next state, the action to apply, and a fixed four-key metric schema. Every
            value is a zero placeholder on the iterations that do not learn, so the pytree
            returned to ``lax.scan`` is identical on every iteration. The gradient norms
            are worth reporting here because both optimizers clip at a global norm of 1.
        """
        action_dim = state.last_action.shape[0]
        actor, _ = _networks(action_dim, self.config.HIDDEN_SIZE)
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
            learner_state: GACAlgorithmState,
            replay_state: ReplayBufferState,
            keys: tuple[jax.Array, ...],
        ) -> tuple[GACAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
            return self._learn(learner_state, replay_state, buffer, keys[0], action_dim)

        def select_action(
            learner_state: GACAlgorithmState,
            observation: jax.Array,
            _step_index: jax.Array,
            keys: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            action = actor_sample(
                actor,
                learner_state.actor_state.params,
                observation,
                keys[0],
                learner_state.hypers.INFERENCE_SIGMA_MIN,
                n=1,
            )[0][0].astype(state.last_action.dtype)
            return action, {}

        hypers = state.algorithm_state.hypers
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
        )

    def _learn(
        self,
        learner_state: GACAlgorithmState,
        replay_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        action_dim: int,
    ) -> tuple[GACAlgorithmState, ReplayBufferState, dict[str, jax.Array]]:
        """Take one critic and one actor gradient step on a replay minibatch.

        The actor's targets are re-proposed against the critic that has just moved, which
        is what makes this a policy-improvement step rather than a fit to a stale ranking.
        """
        config = self.config
        hypers = learner_state.hypers
        actor, critic = _networks(action_dim, config.HIDDEN_SIZE)
        sample_key, next_key, proposal_key = jax.random.split(key, 3)
        obs, actions, rewards, next_obs, discounts = buffer.sample(replay_state, sample_key, config.BATCH_SIZE)

        next_actions = _select_best_next_action(
            learner_state.critic_state.params,
            learner_state.actor_state.params,
            next_obs,
            jax.random.split(next_key, config.BATCH_SIZE),
            config.NUM_RAND_ACTIONS,
            action_dim,
            hypers.INFERENCE_SIGMA_MIN,
            critic=critic,
            actor=actor,
        )

        critic_state = with_hyperparams(learner_state.critic_state, learning_rate=hypers.LEARNING_RATE)

        def _critic_loss_fn(params: VariableDict) -> jax.Array:
            loss, _ = gac_critic_loss(
                params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                next_actions,
                hypers.H_REGULARIZATION,
                critic=critic,
            )
            return loss

        critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        top_actions_batch = _propose_and_rank_topk(
            critic_state.params,
            learner_state.actor_state.params,
            obs,
            jax.random.split(proposal_key, config.BATCH_SIZE),
            config.NUM_SAMPLES,
            config.UNIFORM_WEIGHT,
            config.ACTOR_PERCENTILE,
            action_dim,
            hypers.TRAINING_SIGMA_MIN,
            critic=critic,
            actor=actor,
        )

        actor_state = with_hyperparams(learner_state.actor_state, learning_rate=hypers.ACTOR_LEARNING_RATE)

        def actor_loss_fn(params: VariableDict) -> jax.Array:
            return gac_actor_loss(
                params,
                obs,
                top_actions_batch,
                hypers.ENTROPY_WEIGHT,
                hypers.TRAINING_SIGMA_MIN,
                actor=actor,
            )

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        return replace(learner_state, actor_state=actor_state, critic_state=critic_state), replay_state, {
            "critic_loss": critic_loss.astype(jnp.float32),
            "actor_loss": actor_loss.astype(jnp.float32),
            "critic_grad_norm": tree_norm(critic_grads).astype(jnp.float32),
            "actor_grad_norm": tree_norm(actor_grads).astype(jnp.float32),
        }
