"""PPO as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

:class:`PPOAgent` is the port. ``make_train`` below it is the private training loop the
port replaces, and it is still here because ``projects/missingness-rl`` calls it from an
independent repository: ``missingness_rl/experiment_execution.py`` reads
``out["metrics"]["returned_episode_returns"]`` and its own environment-``info`` metrics, and
``experiment_evaluation.py`` reads ``out["runner_state"][0].params``, out of the result. That
repository resolves ``rl-agents`` as ``{ workspace = true }`` from this worktree, so it
breaks the instant either symbol disappears -- not at some later merge.
``projects/generative-replay`` and ``projects/learning-dynamics`` no longer call it; both
already drive :class:`PPOAgent` through :func:`rl_components.loop.run`.

**Removal condition:** once ``missingness-rl`` ports too. That port needs a metrics path for
the environment ``info`` its observation wrapper emits, which
:func:`rl_components.loop.run` does not forward today, so it is a loop change rather than a
project-side rewrite. Five ``examples/`` scripts and two ``tests/performance`` benches inside
this repository also still call ``make_train`` and must be re-seated in the same change.

``PPOConfig.GAMMA`` and ``PPOHypers.GAMMA`` survive only for that legacy path. The discount
belongs to :func:`rl_components.loop.run`, which puts a per-transition coefficient on
``Timestep.discount``; :class:`PPOAgent` reads that coefficient and never reads ``GAMMA``.
Two sources of truth for the discount is how the bootstrap defect this port exists to fix
recurs.

Two things separate this port from a re-seating, and both are recorded in §4.5 of
``docs/specs/agent-port-and-training-loop.md``:

- **Rollout slots are completion-indexed.** See :class:`PPOTransition`. The pre-port layout
  put reward and done in the slot the action was chosen in, which a single forward
  ``lax.scan`` pass cannot produce once the loop delivers them with the observation that
  follows the action.
- **The bootstrap coefficient and the trajectory break are separate quantities.** The
  pre-port ``_calculate_gae`` drove both from one fused ``not_done``, so a truncation either
  lost the bootstrap it was entitled to or leaked a trace across the boundary. See
  :func:`ppo_gae`.

The losses are module-level functions rather than closures inside ``make_train``, so the
tests gate the real arithmetic instead of a copy of it.
"""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple, Protocol, TypedDict, cast

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.agent_protocol import AgentStep
from rl_components.env_protocol import EnvSpec
from rl_components.gym_env import ContinuousActionSpace, DiscreteActionSpace, GymEnv
from rl_components.networks import ActorCritic, ContinuousActorCritic
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep
from rl_components.types import PPOConfig

from rl_agents.continuous_actions import continuous_action_dim


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


class _ObservationNormState(NamedTuple):
    observation_count: jax.Array
    mean: jax.Array
    m2: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: object
    last_obs: jax.Array
    obs_norm_state: _ObservationNormState
    rng: jax.Array


def _is_discrete_action_space(action_space: object) -> bool:
    return hasattr(action_space, "n")


def _continuous_action_dim(action_space: ContinuousActionSpace) -> int:
    if len(action_space.shape) != 1:
        raise ValueError(f"continuous PPO expects a flat action shape, got {action_space.shape}")
    return action_space.shape[0]


def _sum_action_event_terms(values: object, *, is_continuous: bool) -> jax.Array:
    values_array = jnp.asarray(values)
    if not is_continuous:
        return values_array
    return jnp.sum(values_array, axis=-1)


def _empty_observation_norm_state(obs: jax.Array) -> _ObservationNormState:
    obs_array = jnp.asarray(obs, dtype=jnp.float32)
    zeros = jnp.zeros_like(obs_array)
    return _ObservationNormState(observation_count=jnp.array(0.0, dtype=jnp.float32), mean=zeros, m2=zeros)


def _update_observation_norm_state(state: _ObservationNormState, obs: jax.Array) -> _ObservationNormState:
    obs_array = jnp.asarray(obs, dtype=jnp.float32)
    observation_count = state.observation_count + jnp.array(1.0, dtype=jnp.float32)
    delta = obs_array - state.mean
    mean = state.mean + delta / observation_count
    delta2 = obs_array - mean
    m2 = state.m2 + delta * delta2
    return _ObservationNormState(observation_count=observation_count, mean=mean, m2=m2)


def _init_observation_norm_state(obs: jax.Array) -> _ObservationNormState:
    return _update_observation_norm_state(_empty_observation_norm_state(obs), obs)


def _normalize_observation(state: _ObservationNormState, obs: jax.Array, *, eps: jax.Array | float, clip: jax.Array | float) -> jax.Array:
    obs_array = jnp.asarray(obs, dtype=jnp.float32)
    variance = jnp.where(
        state.observation_count > 0.0,
        state.m2 / state.observation_count,
        jnp.ones_like(state.mean),
    )
    normalized_obs = (obs_array - state.mean) / jnp.sqrt(variance + jnp.asarray(eps, dtype=obs_array.dtype))
    clip_value = jnp.asarray(clip, dtype=obs_array.dtype)
    return jnp.clip(normalized_obs, -clip_value, clip_value)


def _maybe_update_observation_norm_state(state: _ObservationNormState, obs: jax.Array, *, enabled: bool) -> _ObservationNormState:
    if not enabled:
        return state
    return _update_observation_norm_state(state, obs)


def _maybe_normalize_observation(
    obs: jax.Array,
    state: _ObservationNormState,
    *,
    eps: jax.Array | float,
    clip: jax.Array | float,
    enabled: bool,
) -> jax.Array:
    if not enabled:
        return obs
    return _normalize_observation(state, obs, eps=eps, clip=clip)


@chex_struct(frozen=True)
class PPOTransition:
    """One completion-indexed rollout slot, or a stacked window of them.

    Slot ``i`` records the transition **closed** at loop iteration ``i``, not the one
    opened there. ``lax.scan`` forces this: a scan body at iteration ``i`` can only emit
    values it holds at ``i``, so a layout referring to ``ts_{i+1}`` is not implementable in
    a single forward pass. The first four fields are what the agent carried into the
    iteration; the next three arrive with the incoming
    :class:`~rl_components.timestep.Timestep`.

    Attributes:
        obs: ``s_{i-1}``, already normalized when the agent normalizes, because the loss
            reruns the network on exactly what the action was selected from.
        action: ``a_{i-1}``, the action that opened the transition.
        log_prob: The behaviour policy's log-probability of ``action``, for the ratio.
        value: ``V(s_{i-1})`` as estimated when that action was selected.
        reward: The reward ``ts_i`` closed the transition with, after ``REWARD_SCALE``.
        discount: ``ts_i.discount``: the bootstrap coefficient, ``0`` on termination and
            ``gamma`` on continuation and truncation alike.
        episode_end: ``ts_i.episode_end``: the trajectory break, true on termination and
            truncation alike. It cuts the GAE trace and does nothing else.
        bootstrap_value: ``V(ts_i.bootstrap_observation)``, computed at iteration ``i``
            while the true state reached is still in hand. Storing it per slot is what
            decouples advantage accumulation from value propagation; recovering it by
            carrying the following slot's ``value`` backward through the reverse scan is
            the cross-episode leak, because at a boundary that value is ``V(s_0_new)``.
        valid: ``step_index > 0``. False on exactly one iteration of the whole run --
            global ``t = 0``, the only iteration that closes no transition -- so the first
            rollout holds ``NUM_STEPS - 1`` valid slots and every later one holds
            ``NUM_STEPS``.
    """

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    discount: jax.Array
    episode_end: jax.Array
    bootstrap_value: jax.Array
    valid: jax.Array


def ppo_gae(rollout: PPOTransition, gae_lambda: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over one completion-indexed rollout window.

    The two boundary quantities drive different halves of the recurrence, and separating
    them is the correctness half of this port: the pre-port code drove both from a single
    fused ``not_done`` and so could not express a truncation at all.

    - ``discount`` is the bootstrap coefficient of the TD residual. It is ``0`` only where
      the bootstrap is genuinely dead, so a truncated transition keeps the estimate of the
      state it actually reached.
    - ``episode_end`` cuts the trace, and is true on truncation as well, so no advantage
      accumulates across a boundary.

    ``discount * gae_lambda * (1 - episode_end)`` equals the textbook
    ``gamma * lam * (1 - done)`` in all three cases, because wherever ``discount`` differs
    from ``gamma`` the ``(1 - episode_end)`` factor is already zero. That equality holds
    under both ``truncation_policy`` values, which is why PPO needs no ``gamma`` of its own.

    Args:
        rollout: One rollout window, leading axis ``NUM_STEPS``.
        gae_lambda: The trace-decay parameter, traced so a sweep can batch over it.

    Returns:
        The advantages and the value targets, each shaped like ``rollout.value``.
    """

    def _advance(gae: jax.Array, transition: PPOTransition) -> tuple[jax.Array, jax.Array]:
        continues = jnp.asarray(1.0, jnp.float32) - transition.episode_end.astype(jnp.float32)
        delta = transition.reward + transition.discount * transition.bootstrap_value - transition.value
        gae = delta + transition.discount * gae_lambda * continues * gae
        return gae, gae

    _, advantages = jax.lax.scan(_advance, jnp.zeros((), jnp.float32), rollout, reverse=True)
    return advantages, advantages + rollout.value


class PPOPolicy(Protocol):
    """The three things PPO needs from whatever distribution its network head returns."""

    def sample(self, *, seed: jax.Array) -> jax.Array: ...

    def log_prob(self, value: jax.Array) -> jax.Array: ...

    def entropy(self) -> jax.Array: ...


type PPOApplyFn = Callable[..., tuple[PPOPolicy, jax.Array]]
"""An actor-critic module's ``apply``, as reached through ``TrainState.apply_fn``."""


def _masked_mean(values: jax.Array, valid: jax.Array) -> jax.Array:
    """Average ``values`` over the valid slots alone.

    A minibatch can hold the single invalid slot of the whole run -- global ``t = 0``, the
    one iteration that closes no transition -- and after a shuffle it could in principle
    hold nothing else, so the denominator is floored at one rather than trusted.
    """
    mask = valid.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, values.dtype))


def ppo_value_loss(
    value: jax.Array,
    old_value: jax.Array,
    targets: jax.Array,
    valid: jax.Array,
    *,
    clip_eps: jax.Array,
) -> jax.Array:
    """Clipped value loss over one minibatch: the worse of the raw and clipped errors.

    Args:
        value: Values the network predicts now, under the parameters being differentiated.
        old_value: Values stored when the actions were selected, the clip's centre.
        targets: GAE value targets.
        valid: Which slots close a real transition.
        clip_eps: Trust-region half-width, shared with the policy loss.

    Returns:
        The scalar loss.
    """
    value_clipped = old_value + jnp.clip(value - old_value, -clip_eps, clip_eps)
    losses = jnp.square(value - targets)
    losses_clipped = jnp.square(value_clipped - targets)
    return 0.5 * _masked_mean(jnp.maximum(losses, losses_clipped), valid)


def ppo_policy_loss(
    log_prob: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    valid: jax.Array,
    *,
    clip_eps: jax.Array,
) -> jax.Array:
    """Clipped surrogate objective over one minibatch, as a loss to minimize.

    The advantages are standardized across the whole minibatch rather than across its valid
    slots. The invalid slot's advantage is exactly zero by construction -- it closes no
    transition, so its reward, discount and carried value are all zero-primed and its TD
    residual and trace factor both vanish -- so it shifts the mean by one zero in one
    minibatch of the first update and nothing after that.

    Args:
        log_prob: Log-probabilities under the parameters being differentiated.
        old_log_prob: Log-probabilities of the behaviour policy that acted.
        advantages: GAE advantages for the minibatch.
        valid: Which slots close a real transition.
        clip_eps: Trust-region half-width.

    Returns:
        The scalar loss.
    """
    ratio = jnp.exp(log_prob - old_log_prob)
    standardized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    unclipped = ratio * standardized
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * standardized
    return -_masked_mean(jnp.minimum(unclipped, clipped), valid)


def ppo_loss(
    params: VariableDict,
    transitions: PPOTransition,
    advantages: jax.Array,
    targets: jax.Array,
    *,
    apply_fn: PPOApplyFn,
    is_continuous: bool,
    clip_eps: jax.Array,
    vf_coef: jax.Array,
    ent_coef: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """PPO's total objective over one minibatch, with its three components as auxiliaries.

    Reachable from outside the agent so the tests gate the real arithmetic rather than a
    copy of it: while these expressions were closures inside ``make_train``, the tests that
    covered them re-implemented every term on local arrays and so agreed only with
    themselves.

    Every component takes a masked mean, so the single invalid slot of the run contributes
    nothing to any of the three.

    Args:
        params: Network parameters. Differentiate with respect to this argument only.
        transitions: One minibatch of completion-indexed rollout slots.
        advantages: GAE advantages for those slots.
        targets: GAE value targets for those slots.
        apply_fn: The actor-critic's ``apply``, static because it is a Python callable.
        is_continuous: Whether the policy is a diagonal distribution over an action vector,
            whose per-dimension log-probabilities and entropies must be summed. A static
            Python bool, recovered from the rank of the stored action.
        clip_eps: Trust-region half-width, shared by both clipped terms.
        vf_coef: Weight on the value loss.
        ent_coef: Weight on the entropy bonus, which is subtracted.

    Returns:
        The scalar total loss, and the ``(value_loss, policy_loss, entropy)`` triple. The
        third is the entropy itself, not a loss: it enters the total negated.
    """
    policy, value = apply_fn(params, transitions.obs)
    log_prob = _sum_action_event_terms(policy.log_prob(transitions.action), is_continuous=is_continuous)

    value_loss = ppo_value_loss(value, transitions.value, targets, transitions.valid, clip_eps=clip_eps)
    policy_loss = ppo_policy_loss(log_prob, transitions.log_prob, advantages, transitions.valid, clip_eps=clip_eps)
    entropy = _masked_mean(_sum_action_event_terms(policy.entropy(), is_continuous=is_continuous), transitions.valid)

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, (value_loss, policy_loss, entropy)


class PPOTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


@chex_struct(frozen=True)
class PPOHypers:
    """The hyperparameters ``train`` reads as traced values.

    These ride ``train``'s argument list rather than ``config`` so a sweep can
    ``jax.vmap`` one compiled kernel across a batch of arms. Everything here is
    read only in arithmetic, never to size an array or take a Python branch --
    those stay on ``config``.
    """

    LR: jax.Array
    GAMMA: jax.Array
    GAE_LAMBDA: jax.Array
    CLIP_EPS: jax.Array
    ENT_COEF: jax.Array
    VF_COEF: jax.Array
    MAX_GRAD_NORM: jax.Array
    REWARD_SCALE: jax.Array
    OBS_NORM_EPS: jax.Array
    OBS_NORM_CLIP: jax.Array


def ppo_hypers(config: PPOConfig) -> PPOHypers:
    return PPOHypers(
        LR=jnp.asarray(config.LR, jnp.float32),
        GAMMA=jnp.asarray(config.GAMMA, jnp.float32),
        GAE_LAMBDA=jnp.asarray(config.GAE_LAMBDA, jnp.float32),
        CLIP_EPS=jnp.asarray(config.CLIP_EPS, jnp.float32),
        ENT_COEF=jnp.asarray(config.ENT_COEF, jnp.float32),
        VF_COEF=jnp.asarray(config.VF_COEF, jnp.float32),
        MAX_GRAD_NORM=jnp.asarray(config.MAX_GRAD_NORM, jnp.float32),
        REWARD_SCALE=jnp.asarray(config.REWARD_SCALE, jnp.float32),
        OBS_NORM_EPS=jnp.asarray(config.OBS_NORM_EPS, jnp.float32),
        OBS_NORM_CLIP=jnp.asarray(config.OBS_NORM_CLIP, jnp.float32),
    )


def _ppo_tx(learning_rate: jax.Array, max_grad_norm: jax.Array) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )


@chex_struct(frozen=True)
class PPOAgentState:
    """Everything :class:`PPOAgent` carries between loop iterations.

    Attributes:
        train_state: Actor-critic parameters, optimizer state, and the static ``apply_fn``
            through which ``step`` reaches the network.
        obs_norm_state: Running observation statistics. Present, and inert, when
            normalization is off, so the scan carry does not change shape with the flag.
        rollout: The rollout window under construction, leading axis ``NUM_STEPS``. Slot
            ``step_index % NUM_STEPS`` is written every iteration and the window is
            consumed on the iteration where that index reaches ``NUM_STEPS - 1``.
        last_obs: The observation the pending action was selected from, stored already
            normalized because the loss reruns the network on exactly that input.
        last_action: The action that opened the pending transition. Its rank is also where
            ``step`` recovers whether the policy is continuous, since ``step`` never sees a
            spec.
        last_log_prob: That action's log-probability under the policy that chose it, which
            is the denominator of PPO's ratio and must be the behaviour policy's.
        last_value: The critic's estimate of ``last_obs`` at the time the action was chosen.
        key: PRNG key for sampling actions and shuffling minibatches.
        hypers: Swept hyperparameters, traced so a batch of arms shares one kernel.

    The four ``last_*`` fields are zero-primed at ``init``, which sees no observation. Only
    rollout slot ``0`` of the first window ever reads the primed values, and that slot is
    the one the ``valid`` mask removes.
    """

    train_state: TrainState
    obs_norm_state: _ObservationNormState
    rollout: PPOTransition
    last_obs: jax.Array
    last_action: jax.Array
    last_log_prob: jax.Array
    last_value: jax.Array
    key: chex.PRNGKey
    hypers: PPOHypers


class PPOAgent:
    """Proximal policy optimization over a rollout window the agent fills itself.

    PPO is on-policy, so what replaces the replay buffer of the off-policy agents is a
    fixed ``NUM_STEPS`` window written one slot per loop iteration. The window is consumed
    -- GAE, then ``UPDATE_EPOCHS`` passes over ``NUM_MINIBATCHES`` shuffled minibatches --
    on the iteration whose slot index is ``NUM_STEPS - 1``, under a ``lax.cond`` whose
    predicate reads ``step_index`` alone. That keeps the predicate unbatched under
    ``jax.vmap`` over seeds, so the cond stays a cond and the update does not execute on
    every step of every seed.

    Because the window is indexed by the iteration at which a transition *completes*
    (see :class:`PPOTransition`), the first window holds ``NUM_STEPS - 1`` valid slots and
    every window after it holds ``NUM_STEPS``. There is no per-rollout boundary artefact:
    ``step_index`` counts iterations of the whole run, so exactly one slot of one window is
    ever invalid.

    Observation normalization keeps the pre-port statistics. The running mean and variance
    are updated from ``timestep.observation`` and the same, just-updated statistics then
    normalize both that observation and ``timestep.bootstrap_observation``. So the state
    that normalizes an observation always includes that observation, which is what the
    pre-port loop did by updating from ``obsv`` at the end of an iteration and normalizing
    the identical value as ``last_obs`` at the start of the next. The final
    ``observation_count`` is one lower than the pre-port count for the same budget, because
    a run of ``N`` iterations closes ``N - 1`` transitions and the final iteration's next
    state never reaches the agent.

    The agent owns no environment, no ``lax.scan`` over the horizon and no discount:
    :func:`rl_components.loop.run` supplies the horizon and ``gamma``, and the per-transition
    bootstrap coefficient reaches GAE through ``Timestep.discount``.
    """

    def __init__(self, config: PPOConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar. ``GAMMA`` is not read at all.

        Raises:
            ValueError: If ``REWARD_SCALE`` is not finite and positive, which would scale
                every stored reward to zero, infinity or NaN in silence.
        """
        if not math.isfinite(config.REWARD_SCALE) or config.REWARD_SCALE <= 0.0:
            raise ValueError(f"REWARD_SCALE must be finite and > 0, got {config.REWARD_SCALE!r}")
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> PPOAgentState:
        """Build the network, the optimizer, the empty rollout window and the primed slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description, and the only place the observation shape and
                the action space come from.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space that is not a
                one-dimensional vector normalized to ``[-1, 1]``, which the tanh-squashed
                continuous head requires.
        """
        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_shape = tuple(spec.action_shape)
        action_dtype = jnp.dtype(spec.action_dtype)
        if spec.num_actions is None:
            network = ContinuousActorCritic(continuous_action_dim(spec), activation="tanh")
        else:
            network = ActorCritic(spec.num_actions, activation="tanh")

        params_key, carry_key = jax.random.split(key)
        zero_obs = jnp.zeros(observation_shape, dtype=observation_dtype)
        hypers = ppo_hypers(self.config)
        # Normalization casts to float32; without it the raw observation rides through, so
        # the stored dtype follows the flag or the first slot write is a silent promotion.
        stored_obs_dtype = jnp.dtype(jnp.float32) if self.config.NORMALIZE_OBSERVATIONS else observation_dtype
        num_steps = self.config.NUM_STEPS
        scalars = jnp.zeros((num_steps,), dtype=jnp.float32)

        return PPOAgentState(
            train_state=TrainState.create(
                apply_fn=network.apply,
                params=network.init(params_key, zero_obs),
                tx=optax.inject_hyperparams(_ppo_tx)(learning_rate=hypers.LR, max_grad_norm=hypers.MAX_GRAD_NORM),
            ),
            obs_norm_state=_empty_observation_norm_state(zero_obs),
            rollout=PPOTransition(
                obs=jnp.zeros((num_steps, *observation_shape), dtype=stored_obs_dtype),
                action=jnp.zeros((num_steps, *action_shape), dtype=action_dtype),
                log_prob=scalars,
                value=scalars,
                reward=scalars,
                discount=scalars,
                episode_end=jnp.zeros((num_steps,), dtype=jnp.bool_),
                bootstrap_value=scalars,
                valid=jnp.zeros((num_steps,), dtype=jnp.bool_),
            ),
            last_obs=jnp.zeros(observation_shape, dtype=stored_obs_dtype),
            last_action=jnp.zeros(action_shape, dtype=action_dtype),
            last_log_prob=jnp.zeros((), jnp.float32),
            last_value=jnp.zeros((), jnp.float32),
            key=carry_key,
            hypers=hypers,
        )

    def step(
        self,
        state: PPOAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[PPOAgentState, jax.Array]:
        """Close the pending transition into a rollout slot, update if the window is full, then act.

        The order is normative. The closed slot's ``bootstrap_value`` is read from
        ``timestep.bootstrap_observation`` and not ``timestep.observation``: at a boundary
        the latter is the post-reset observation, and valuing it as the state the transition
        reached is the cross-episode leak this port removes.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so the slot
                it writes is marked invalid and every loss masks it out.

        Returns:
            The next state, the action to apply, and a fixed four-key metric schema. Each
            entry is a zero placeholder on the iterations that do not update, so the pytree
            returned to ``lax.scan`` is identical on every iteration.
        """
        hypers = state.hypers
        is_continuous = state.last_action.ndim == 1
        num_steps = self.config.NUM_STEPS
        carry_key, action_key, update_key = jax.random.split(state.key, 3)

        obs_norm_state = _maybe_update_observation_norm_state(
            state.obs_norm_state, timestep.observation, enabled=self.config.NORMALIZE_OBSERVATIONS
        )
        _, bootstrap_value = state.train_state.apply_fn(
            state.train_state.params,
            self._normalize(timestep.bootstrap_observation, obs_norm_state, hypers),
        )

        closed = PPOTransition(
            obs=state.last_obs,
            action=state.last_action,
            log_prob=state.last_log_prob,
            value=state.last_value,
            reward=(timestep.reward * hypers.REWARD_SCALE).astype(jnp.float32),
            discount=timestep.discount,
            episode_end=timestep.episode_end,
            bootstrap_value=jnp.asarray(bootstrap_value, jnp.float32),
            valid=step_index > 0,
        )
        slot = jnp.remainder(step_index, num_steps)
        rollout = jax.tree_util.tree_map(lambda window, value: window.at[slot].set(value), state.rollout, closed)

        train_state, losses = jax.lax.cond(
            slot == num_steps - 1,
            lambda: self._update(state.train_state, rollout, update_key, hypers, is_continuous=is_continuous),
            lambda: (
                state.train_state,
                (
                    jnp.zeros((), jnp.float32),
                    jnp.zeros((), jnp.float32),
                    jnp.zeros((), jnp.float32),
                    jnp.zeros((), jnp.float32),
                ),
            ),
        )

        normalized_obs = self._normalize(timestep.observation, obs_norm_state, hypers)
        policy, value = train_state.apply_fn(train_state.params, normalized_obs)
        action = policy.sample(seed=action_key).astype(state.last_action.dtype)
        log_prob = _sum_action_event_terms(policy.log_prob(action), is_continuous=is_continuous)
        total_loss, value_loss, policy_loss, entropy = losses

        return AgentStep(
            state=PPOAgentState(
                train_state=train_state,
                obs_norm_state=obs_norm_state,
                rollout=rollout,
                last_obs=normalized_obs,
                last_action=action,
                last_log_prob=jnp.asarray(log_prob, jnp.float32),
                last_value=jnp.asarray(value, jnp.float32),
                key=carry_key,
                hypers=hypers,
            ),
            action=action,
            metrics={
                "total_loss": total_loss,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "entropy": entropy,
            },
        )

    def _normalize(self, observation: jax.Array, obs_norm_state: _ObservationNormState, hypers: PPOHypers) -> jax.Array:
        """Normalize one observation against the current statistics, or pass it through."""
        return _maybe_normalize_observation(
            observation,
            obs_norm_state,
            eps=hypers.OBS_NORM_EPS,
            clip=hypers.OBS_NORM_CLIP,
            enabled=self.config.NORMALIZE_OBSERVATIONS,
        )

    def _update(
        self,
        train_state: TrainState,
        rollout: PPOTransition,
        key: chex.PRNGKey,
        hypers: PPOHypers,
        *,
        is_continuous: bool,
    ) -> tuple[TrainState, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        """Run GAE over the finished window, then ``UPDATE_EPOCHS`` passes of minibatch ascent.

        The advantages and targets are computed once and closed over: they do not depend on
        the parameters being updated, so recomputing them per epoch would be both wasted
        work and a different algorithm. The observation statistics are deliberately absent
        here as well -- ``rollout.obs`` was normalized when it was collected, and
        re-normalizing it against statistics that have since moved would make the ratio's
        denominator no longer the behaviour policy's.
        """
        advantages, targets = ppo_gae(rollout, hypers.GAE_LAMBDA)

        def _minibatch(
            train_state: TrainState,
            batch: tuple[PPOTransition, jax.Array, jax.Array],
        ) -> tuple[TrainState, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            transitions, batch_advantages, batch_targets = batch
            (loss, (value_loss, policy_loss, entropy)), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                train_state.params,
                transitions,
                batch_advantages,
                batch_targets,
                apply_fn=train_state.apply_fn,
                is_continuous=is_continuous,
                clip_eps=hypers.CLIP_EPS,
                vf_coef=hypers.VF_COEF,
                ent_coef=hypers.ENT_COEF,
            )
            # The transformation is a static field and cannot hold a traced value, so the
            # swept rate and clip norm are pushed into the optimizer state each step.
            opt_state = train_state.opt_state
            train_state = train_state.replace(
                opt_state=opt_state._replace(
                    hyperparams={
                        **opt_state.hyperparams,
                        "learning_rate": hypers.LR,
                        "max_grad_norm": hypers.MAX_GRAD_NORM,
                    }
                ),
            )
            return train_state.apply_gradients(grads=grads), (loss, value_loss, policy_loss, entropy)

        def _epoch(
            carry: tuple[TrainState, chex.PRNGKey],
            unused: None,
        ) -> tuple[tuple[TrainState, chex.PRNGKey], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            del unused
            train_state, epoch_key = carry
            epoch_key, permutation_key = jax.random.split(epoch_key)
            permutation = jax.random.permutation(permutation_key, self.config.NUM_STEPS)
            shuffled = jax.tree_util.tree_map(
                lambda leaf: jnp.take(leaf, permutation, axis=0).reshape((self.config.NUM_MINIBATCHES, -1, *leaf.shape[1:])),
                (rollout, advantages, targets),
            )
            train_state, losses = jax.lax.scan(_minibatch, train_state, shuffled)
            return (train_state, epoch_key), losses

        (train_state, _), losses = jax.lax.scan(_epoch, (train_state, key), None, self.config.UPDATE_EPOCHS)
        total_loss, value_loss, policy_loss, entropy = losses
        return train_state, (
            jnp.mean(total_loss).astype(jnp.float32),
            jnp.mean(value_loss).astype(jnp.float32),
            jnp.mean(policy_loss).astype(jnp.float32),
            jnp.mean(entropy).astype(jnp.float32),
        )


def make_train(
    config: PPOConfig,
    env: GymEnv[DiscreteActionSpace | ContinuousActionSpace],
    env_params: object | None = None,
) -> Callable[[jax.Array, PPOHypers | None], PPOTrainOutput]:
    if not math.isfinite(config.REWARD_SCALE) or config.REWARD_SCALE <= 0.0:
        raise ValueError(f"REWARD_SCALE must be finite and > 0, got {config.REWARD_SCALE!r}")

    normalize_observations = config.NORMALIZE_OBSERVATIONS

    def train(rng: jax.Array, hypers: PPOHypers | None = None) -> PPOTrainOutput:
        if hypers is None:
            hypers = ppo_hypers(config)
        reward_scale = jnp.asarray(hypers.REWARD_SCALE, dtype=jnp.float32)

        # INIT NETWORK
        action_space = env.action_space(env_params)
        continuous_actions = not _is_discrete_action_space(action_space)
        if continuous_actions:
            continuous_action_space = cast(ContinuousActionSpace, action_space)
            network = ContinuousActorCritic(_continuous_action_dim(continuous_action_space), activation="tanh")
        else:
            discrete_action_space = cast(DiscreteActionSpace, action_space)
            network = ActorCritic(discrete_action_space.n, activation="tanh")
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        tx = optax.inject_hyperparams(_ppo_tx)(learning_rate=hypers.LR, max_grad_norm=hypers.MAX_GRAD_NORM)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        obs_norm_state = _init_observation_norm_state(obsv)

        # TRAIN LOOP
        def _update_step(runner_state: RunnerState, unused: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            train_state, env_state, last_obs, obs_norm_state, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, unused: jax.Array) -> tuple[RunnerState, Transition]:
                train_state, env_state, last_obs, obs_norm_state, rng = runner_state
                normalized_obs = _maybe_normalize_observation(
                    last_obs,
                    obs_norm_state,
                    eps=hypers.OBS_NORM_EPS,
                    clip=hypers.OBS_NORM_CLIP,
                    enabled=normalize_observations,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                policy, value = network.apply(train_state.params, normalized_obs)
                action = policy.sample(seed=_rng)
                log_prob = _sum_action_event_terms(policy.log_prob(action), is_continuous=continuous_actions)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
                scaled_reward = reward * reward_scale
                obs_norm_state = _maybe_update_observation_norm_state(obs_norm_state, obsv, enabled=normalize_observations)
                transition = Transition(done, action, value, scaled_reward, log_prob, normalized_obs, info)
                runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv, obs_norm_state=obs_norm_state, rng=rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.NUM_STEPS)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, obs_norm_state, rng = runner_state
            normalized_last_obs = _maybe_normalize_observation(
                last_obs,
                obs_norm_state,
                eps=hypers.OBS_NORM_EPS,
                clip=hypers.OBS_NORM_CLIP,
                enabled=normalize_observations,
            )
            _, last_val = network.apply(train_state.params, normalized_last_obs)

            def _calculate_gae(traj_batch: Transition, last_val: jax.Array) -> tuple[jax.Array, jax.Array]:
                def _get_advantages(gae_and_next_value: tuple[jax.Array, jax.Array], transition: Transition) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    not_done = 1.0 - done
                    delta = reward + hypers.GAMMA * next_value * not_done - value
                    gae = delta + hypers.GAMMA * hypers.GAE_LAMBDA * not_done * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(
                update_state: tuple[TrainState, Transition, jax.Array, jax.Array, jax.Array],
                unused: jax.Array,
            ) -> tuple[tuple[TrainState, Transition, jax.Array, jax.Array, jax.Array], Any]:
                def _update_minbatch(
                    train_state: TrainState,
                    batch_info: tuple[Transition, jax.Array, jax.Array],
                ) -> tuple[TrainState, tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]]:
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(
                        params: VariableDict,
                        traj_batch: Transition,
                        advantages: jax.Array,
                        targets: jax.Array,
                    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
                        # RERUN NETWORK
                        policy, value = network.apply(params, traj_batch.obs)
                        log_prob = _sum_action_event_terms(policy.log_prob(traj_batch.action), is_continuous=continuous_actions)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-hypers.CLIP_EPS, hypers.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE POLICY LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        loss_pc1 = ratio * gae
                        loss_pc2 = (
                            jnp.clip(
                                ratio,
                                1.0 - hypers.CLIP_EPS,
                                1.0 + hypers.CLIP_EPS,
                            )
                            * gae
                        )
                        policy_loss = -jnp.minimum(loss_pc1, loss_pc2).mean()

                        # CALCULATE ENTROPY LOSS
                        entropy_loss = _sum_action_event_terms(policy.entropy(), is_continuous=continuous_actions).mean()

                        loss = policy_loss + hypers.VF_COEF * value_loss - hypers.ENT_COEF * entropy_loss
                        return loss, (value_loss, policy_loss, entropy_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    opt_state = train_state.opt_state
                    train_state = train_state.replace(
                        opt_state=opt_state._replace(
                            hyperparams={**opt_state.hyperparams, "learning_rate": hypers.LR, "max_grad_norm": hypers.MAX_GRAD_NORM}
                        ),
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.NUM_STEPS
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0).reshape([config.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
                    batch,
                )
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, shuffled_batch)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.UPDATE_EPOCHS)
            train_state = update_state[0]
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj_batch.info)
            metric["returned_episode_returns"] = traj_batch.info["returned_episode_returns"].mean()

            runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=last_obs, obs_norm_state=obs_norm_state, rng=update_state[-1])
            return runner_state, metric

        num_updates = config.TOTAL_TIMESTEPS // config.NUM_STEPS
        runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv, obs_norm_state=obs_norm_state, rng=rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train
