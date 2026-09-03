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
"""

from __future__ import annotations

from typing import Callable, NamedTuple, TypedDict, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.gym_env import ContinuousActionSpace, GymEnv
from rl_components.networks import TanhNormalDiag
from rl_components.structs import chex_struct

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════


@chex_struct(frozen=True, kw_only=True)
class GACConfig:
    """Hyperparameters for the Greedy Actor-Critic algorithm.

    Attributes:
        LR: Critic learning rate.
        ACTOR_LR: Actor learning rate.
        BUFFER_SIZE: Capacity of the uniform replay buffer.
        BATCH_SIZE: Minibatch size for both actor and critic updates.
        TOTAL_TIMESTEPS: Total environment steps per training run.
        LEARNING_STARTS: Steps of random exploration before training begins.
        TRAIN_FREQUENCY: Number of environment steps between training updates.
        GAMMA: Discount factor.
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

    LR: float = 3e-4
    ACTOR_LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    TAU: float = 0.005
    NUM_SAMPLES: int = 32
    ACTOR_PERCENTILE: float = 0.1
    UNIFORM_WEIGHT: float = 0.2
    ENTROPY_WEIGHT: float = 0.01
    NUM_RAND_ACTIONS: int = 10
    TRAINING_SIGMA_MIN: float = 5e-3
    INFERENCE_SIGMA_MIN: float = 1e-4
    H_REGULARIZATION: float = 1.0
    HIDDEN_SIZE: int = 256
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


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


def _make_dist(mean: jax.Array, log_std: jax.Array, sigma_min: float) -> TanhNormalDiag:
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
    sigma_min: float,
    n: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Sample ``n`` actions from the actor's policy (via vmap over PRNG splits)."""
    mean, log_std = _actor_apply(actor, params, x)
    dist = _make_dist(mean, log_std, sigma_min)
    keys = jax.random.split(rng, n)
    actions = jax.vmap(lambda k: dist.sample(seed=k))(keys)
    log_probs = jax.vmap(dist.log_prob)(actions)
    return actions, log_probs


def actor_mean_log_prob(
    actor: GACActor,
    params: VariableDict,
    x: jax.Array,
    actions: jax.Array,
    sigma_min: float,
) -> jax.Array:
    """Compute log-probability of given actions under the actor's policy."""
    mean, log_std = _actor_apply(actor, params, x)
    dist = _make_dist(mean, log_std, sigma_min)
    return dist.log_prob(actions)


def actor_entropy(
    actor: GACActor,
    params: VariableDict,
    x: jax.Array,
    sigma_min: float,
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


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class GACTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def _tree_norm(tree: VariableDict) -> jax.Array:
    """Frobenius norm of all leaves in a parameter tree."""
    leaves = jax.tree.leaves(tree)
    norms = [jnp.sqrt(jnp.sum(jnp.square(leaf))) for leaf in leaves]
    return jnp.sum(jnp.array(norms))


def _h_head_l2(critic_params: VariableDict) -> jax.Array:
    """Sum of squared h-head parameters (tiny — ~256 weights)."""
    params_dict = critic_params.get("params", critic_params)
    h_params = params_dict.get("h_head", {})
    if not h_params:
        return jnp.array(0.0)
    sq_sums = [jnp.sum(jnp.square(p)) for p in jax.tree.leaves(h_params)]
    return jnp.sum(jnp.array(sq_sums))


# ═══════════════════════════════════════════════════════════════
# QRC Critic Loss (element-wise, batched via vmap)
# ═══════════════════════════════════════════════════════════════


def _qrc_loss(
    critic_params: VariableDict,
    transition: QRCTransition,
    next_action: jax.Array,
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

    q, h = _critic_apply(GACCritic(), critic_params, obs, action)
    q_prime, _ = _critic_apply(GACCritic(), critic_params, next_obs, next_action)

    target = reward + gamma * q_prime

    sg = jax.lax.stop_gradient
    delta_l = sg(target) - q  # stopped target
    delta_r = target - sg(q)  # stopped q

    # QRC loss: 1/2 * delta_l^2 + sg(h) * delta_r  (h decoupled via stop_grad)
    q_loss = 0.5 * delta_l**2 + sg(h) * delta_r
    h_loss = 0.5 * (sg(delta_l) - h) ** 2

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


def _batch_qrc_loss(
    critic_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    dones: jax.Array,
    next_actions: jax.Array,
    gamma: jax.Array,
    reg_weight: float,
) -> tuple[jax.Array, QRCMetrics]:
    """Vectorised QRC loss over a batch of transitions.

    L2 regularization is applied **only** to the h-head parameters
    (QRC paper §3.2).
    """
    gamma_arr = gamma * (1.0 - dones)

    transitions = jax.vmap(QRCTransition)(obs, actions, rewards, next_obs, gamma_arr)

    losses, metrics = jax.vmap(_qrc_loss, in_axes=(None, 0, 0))(
        critic_params, transitions, next_actions,
    )

    # L2 regularisation on h-head only (QRC paper §3.2)
    h_reg = _h_head_l2(critic_params)

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
    sigma_min: float,
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

    flat_q, _ = jax.vmap(
        lambda s, a: _critic_apply(critic, critic_params, s, a),
    )(flat_obs, flat_proposals)

    q_values = flat_q.reshape(B, num_samples)
    k = max(1, min(int(actor_percentile * num_samples), num_samples))
    _, top_idxs = jax.lax.top_k(q_values, k)
    return jax.vmap(lambda p, idx: p[idx])(proposals, top_idxs)


def _select_best_next_action(
    critic_params: VariableDict,
    actor_params: VariableDict,
    next_obs: jax.Array,       # (B, obs_dim)
    rngs: jax.Array,           # (B, 2) — one PRNG per batch element
    num_rand_actions: int,
    action_dim: int,
    sigma_min: float,
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

    flat_q, _ = jax.vmap(
        lambda s, a: _critic_apply(critic, critic_params, s, a),
    )(flat_obs, flat_proposals)

    q_values = flat_q.reshape(B, N)
    best_idxs = jnp.argmax(q_values, axis=-1)
    return jax.vmap(lambda p, idx: p[idx])(proposals, best_idxs)


# ═══════════════════════════════════════════════════════════════
# Actor Loss
# ═══════════════════════════════════════════════════════════════


def _actor_loss(
    actor_params: VariableDict,
    state_features: jax.Array,
    top_actions: jax.Array,
    entropy_weight: float,
    sigma_min: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Negative log-likelihood on top actions + entropy bonus."""
    act = GACActor(action_dim=top_actions.shape[-1])
    log_prob = actor_mean_log_prob(act, actor_params, state_features, top_actions, sigma_min)
    nll = -jnp.mean(log_prob)
    ent = jnp.sum(actor_entropy(act, actor_params, state_features, sigma_min))
    loss = nll - entropy_weight * ent
    return loss, {"nll": nll, "entropy": ent}


def _batch_actor_loss(
    actor_params: VariableDict,
    state_features: jax.Array,
    top_actions_batch: jax.Array,  # (batch, k, action_dim)
    entropy_weight: float,
    sigma_min: float,
) -> jax.Array:
    """Mean actor loss across a batch of states."""
    losses, _ = jax.vmap(
        lambda sf, ta: _actor_loss(actor_params, sf, ta, entropy_weight, sigma_min),
    )(state_features, top_actions_batch)
    return jnp.mean(losses)


# ═══════════════════════════════════════════════════════════════
# make_train
# ═══════════════════════════════════════════════════════════════


def make_train(
    config: GACConfig,
    env: GymEnv[ContinuousActionSpace],
    env_params: object | None = None,
) -> Callable[[jax.Array], GACTrainOutput]:
    """Construct the GAC training function.

    Returns a callable ``train(rng)`` that runs a full training
    scan and returns final state + metrics.
    """
    action_dim = env.action_space(env_params).shape[0]
    obs_shape = tuple(env.observation_space(env_params).shape)

    def train(rng: jax.Array) -> GACTrainOutput:
        # ── Initialise networks ──────────────────────────────────
        rng, critic_key, actor_key = jax.random.split(rng, 3)

        critic = GACCritic(hidden_size=config.HIDDEN_SIZE)
        init_obs = jnp.zeros(obs_shape)
        init_action = jnp.zeros((action_dim,))
        critic_params = critic.init(critic_key, init_obs, init_action)
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=config.LR, weight_decay=0.1),
            ),
        )

        actor = GACActor(action_dim, hidden_size=config.HIDDEN_SIZE)
        actor_params = actor.init(actor_key, init_obs)
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=config.ACTOR_LR, weight_decay=0.1),
            ),
        )

        # ── Initialise buffer ────────────────────────────────────
        buffer = ReplayBuffer(config.BUFFER_SIZE, obs_shape, (action_dim,), jnp.float32)
        buffer_state = buffer.init()

        # ── Initialise environment ────────────────────────────────
        rng, reset_rng = jax.random.split(rng)
        obsv, env_state = env.reset(reset_rng, env_params)

        # ══════════════════════════════════════════════════════════
        def _update_step(
            runner_state: RunnerState,
            t: jax.Array,
        ) -> tuple[RunnerState, dict[str, jax.Array]]:
            actor_state, critic_state, buffer_state, env_state, last_obs, rng = runner_state

            # ── Action selection ─────────────────────────────────
            rng, action_rng, step_rng = jax.random.split(rng, 3)

            def _random_action() -> jax.Array:
                return jax.random.uniform(action_rng, (action_dim,), minval=-1.0, maxval=1.0)

            def _policy_action() -> jax.Array:
                action, _ = actor_sample(
                    actor, actor_state.params, last_obs, action_rng,
                    config.INFERENCE_SIGMA_MIN, n=1,
                )
                return action[0]

            action = jax.lax.cond(
                t < config.LEARNING_STARTS,
                _random_action,
                _policy_action,
            )

            # ── Environment step ─────────────────────────────────
            obsv, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)

            # ── Add to buffer ────────────────────────────────────
            buffer_state = buffer.add(
                buffer_state,
                last_obs[None, ...],
                action[None, ...],
                reward[None, ...],
                obsv[None, ...],
                done[None, ...],
            )

            # ── Training ─────────────────────────────────────────
            def _do_train(
                actor_state: TrainState,
                critic_state: TrainState,
                buffer_state: ReplayBufferState,
                rng: jax.Array,
            ) -> tuple[TrainState, TrainState, dict[str, jax.Array]]:
                rng, sample_rng, next_rng, proposal_rngs = jax.random.split(rng, 4)

                obs, actions, rewards, next_obs, dones = buffer.sample(
                    buffer_state, sample_rng, config.BATCH_SIZE,
                )
                gamma = jnp.full((config.BATCH_SIZE,), config.GAMMA)

                # ── Critic update ────────────────────────────────
                batch_rngs = jax.random.split(next_rng, config.BATCH_SIZE)
                next_actions = _select_best_next_action(
                    critic_state.params, actor_state.params, next_obs, batch_rngs,
                    config.NUM_RAND_ACTIONS, action_dim, config.INFERENCE_SIGMA_MIN,
                    critic=critic, actor=actor,
                )

                def _critic_loss_fn(params: VariableDict) -> jax.Array:
                    loss, _ = _batch_qrc_loss(
                        params, obs, actions, rewards, next_obs, dones,
                        next_actions, gamma, config.H_REGULARIZATION,
                    )
                    return loss

                critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(critic_state.params)
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                # ── Actor update ─────────────────────────────────
                batch_rngs = jax.random.split(proposal_rngs, config.BATCH_SIZE)
                top_actions_batch = _propose_and_rank_topk(
                    critic_state.params, actor_state.params, obs, batch_rngs,
                    config.NUM_SAMPLES, config.UNIFORM_WEIGHT,
                    config.ACTOR_PERCENTILE, action_dim,
                    config.TRAINING_SIGMA_MIN,
                    critic=critic, actor=actor,
                )

                def _actor_loss_fn(params: VariableDict) -> jax.Array:
                    return _batch_actor_loss(
                        params, obs, top_actions_batch,
                        config.ENTROPY_WEIGHT, config.TRAINING_SIGMA_MIN,
                    )

                actor_loss, actor_grads = jax.value_and_grad(_actor_loss_fn)(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                train_metrics = {
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "critic_grad_norm": _tree_norm(critic_grads),
                    "actor_grad_norm": _tree_norm(actor_grads),
                }
                return actor_state, critic_state, train_metrics

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, train_metrics = jax.lax.cond(
                can_train,
                lambda: _do_train(actor_state, critic_state, buffer_state, rng),
                lambda: (
                    actor_state,
                    critic_state,
                    {
                        "critic_loss": jnp.array(0.0),
                        "actor_loss": jnp.array(0.0),
                        "critic_grad_norm": jnp.array(0.0),
                        "actor_grad_norm": jnp.array(0.0),
                    },
                ),
            )

            runner_state = RunnerState(
                actor_state=actor_state,
                critic_state=critic_state,
                buffer_state=buffer_state,
                env_state=env_state,
                last_obs=obsv,
                rng=rng,
            )
            return runner_state, {**info, **train_metrics}

        # ── Scan ───────────────────────────────────────────────
        runner_state = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            buffer_state=buffer_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
