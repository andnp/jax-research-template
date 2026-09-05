"""Rainbow as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

The agent owns its noisy duelling C51 network, its prioritised replay, its n-step
accumulator and its centered-RMSProp optimizer. It owns no environment, no ``lax.scan``
and no discount: :func:`rl_components.loop.run` supplies the horizon and ``gamma``.

``ADDITIONAL_DISCOUNT`` is deleted rather than renamed. The loop hands ``step`` a
per-step ``Timestep.discount`` of ``{0, gamma}``, and the accumulator multiplies exactly
those coefficients together, so a second 0.99 beside ``gamma = 0.99`` would discount the
n-step return at 0.9801 -- the two-sources-of-truth defect the port exists to remove.

Multi-step returns are the whole reason this agent is the last port. Everything the loop
knows about a boundary is per step; everything replay needs is per window. The
accumulator in between is where the two meet, and it is the only place in the repository
that has to distinguish ``discount`` from ``episode_end``: a termination is erased by the
coefficient itself, while a truncation keeps a nonzero coefficient and must still cut the
window. All of it stays inside the agent -- the loop knows nothing about n-step.

Three things ``step`` needs are spec-derived, and ``step`` never sees the spec -- it
receives only a :class:`~rl_components.timestep.Timestep` -- while the agent object
itself is static under ``jit`` and so cannot be mutated by ``init``. Each is therefore
reachable from the state instead:

- the network, through ``state.train_state.apply_fn``, which Flax marks
  ``pytree_node=False`` and which therefore survives the scan carry as a static field;
- the replay entry's pytree shape, rebuilt inside ``step`` from the buffer state's own
  leaves, because ``jax_replay.per`` stores them keyed by stringified flatten index and
  needs a prototype of that shape to reassemble a sampled batch;
- the atom support, rebuilt from the traced ``V_MIN``/``V_MAX`` in the carried hypers, so
  a ``vmap`` sweep over the endpoints gets one support per arm.

The can-train predicate gates on replay occupancy rather than the DQN family's
``step_index > LEARNING_STARTS``, which is the predicate this agent already had; its
schedules now key off ``step_index``, the count of transitions closed, where the private
loop counted its own ``env_step``.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.distributional import (
    categorical_cross_entropy,
    categorical_expected_value,
    categorical_l2_project,
)
from jax_nn.typed_module import TypedApply
from jax_replay.per import init_per_buffer, per_add, per_sample, per_update_priorities
from jax_replay.types import PERBufferState
from rl_components.agent_protocol import AgentStep
from rl_components.env_protocol import EnvSpec
from rl_components.layers import NoisyLinear
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep
from rl_networks.layers import NatureCNN

from rl_agents.q_networks import infer_nature_observation_layout, prepare_nature_observations

type RainbowApplyFn = Callable[..., jax.Array]
"""The network's ``apply``, as reached through ``TrainState.apply_fn``.

Noisy layers make it an ``apply`` that must be handed a ``noise`` rng on every call, so
no caller of :func:`rainbow_loss` can omit the key.
"""


@chex_struct(frozen=True, kw_only=True)
class RainbowConfig:
    REPLAY_CAPACITY: int = 1_000_000
    MIN_REPLAY_CAPACITY_FRACTION: float = 0.05
    BATCH_SIZE: int = 32
    NUM_ACTION_REPEATS: int = 4
    TARGET_NETWORK_UPDATE_PERIOD_FRAMES: int = 40_000
    LEARN_PERIOD_FRAMES: int = 16
    LEARNING_RATE: float = 0.00025
    OPTIMIZER_EPSILON: float = 0.01 / 32**2
    RMSPROP_DECAY: float = 0.95
    RMSPROP_CENTERED: bool = True
    N_STEP: int = 3
    NUM_ATOMS: int = 51
    V_MIN: float = -10.0
    V_MAX: float = 10.0


@chex_struct(frozen=True, kw_only=True)
class RainbowRuntimeConfig:
    TOTAL_TRAIN_ENV_STEPS: int = 50_000_000
    SEED: int = 42


@chex_struct(frozen=True)
class RainbowHypers:
    """The hyperparameters ``step`` reads as traced values.

    These ride the state pytree rather than the agent object so a sweep can
    ``jax.vmap`` one compiled kernel across a batch of arms. Everything here is
    read only in arithmetic, never to size an array or take a Python branch --
    those stay on ``RainbowConfig``. There is no discount among them: it belongs
    to :func:`rl_components.loop.run`.
    """

    LEARNING_RATE: jax.Array
    OPTIMIZER_EPSILON: jax.Array
    RMSPROP_DECAY: jax.Array
    V_MIN: jax.Array
    V_MAX: jax.Array


def rainbow_hypers(config: RainbowConfig) -> RainbowHypers:
    return RainbowHypers(
        LEARNING_RATE=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        OPTIMIZER_EPSILON=jnp.asarray(config.OPTIMIZER_EPSILON, jnp.float32),
        RMSPROP_DECAY=jnp.asarray(config.RMSPROP_DECAY, jnp.float32),
        V_MIN=jnp.asarray(config.V_MIN, jnp.float32),
        V_MAX=jnp.asarray(config.V_MAX, jnp.float32),
    )


class RainbowTransition(NamedTuple):
    """One replay entry: an n-step transition and the bootstrap coefficient it earned.

    ``discount`` replaces the boolean ``done`` this entry used to store. A boolean says
    only whether the window ended, which is enough to mask a termination and nothing
    else; the coefficient says how far the window actually reached, which is what a
    window cut short by a truncation needs.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    discount: jax.Array


@chex_struct(frozen=True, kw_only=True)
class _NStepAccumulatorState:
    """The pending window of transitions whose n-step return is still incomplete.

    Attributes:
        obs: Observation each pending transition started from.
        action: Action each pending transition took.
        reward: Partially accumulated discounted reward of each pending transition,
            ``sum_{k<m} (prod_{j<k} d) r``, where ``m`` counts the steps folded in so far.
        discount: Cumulative bootstrap coefficient of each pending transition,
            ``prod_{j<m} d``. A boolean cannot stand in for it: a window cut by an episode
            boundary flushes every pending row at a different horizon, so each row leaves
            with a different product.
        size: Number of pending rows.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    discount: jax.Array
    size: jax.Array


class RainbowNatureNetwork(TypedApply[jax.Array], nn.Module):
    action_dim: int
    num_atoms: int
    observation_layout: Literal["hwc", "fhwc"]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = prepare_nature_observations(x, self.observation_layout)
        x = NatureCNN(dtype=self.dtype)(x)
        x = NoisyLinear(features=512, dtype=self.dtype)(x)
        x = nn.relu(x)

        value_logits = NoisyLinear(features=self.num_atoms, dtype=self.dtype)(x)
        value_logits = value_logits.reshape(value_logits.shape[:-1] + (1, self.num_atoms))

        advantage_logits = NoisyLinear(features=self.action_dim * self.num_atoms, dtype=self.dtype)(x)
        advantage_logits = advantage_logits.reshape(
            advantage_logits.shape[:-1] + (self.action_dim, self.num_atoms)
        )
        centered_advantage_logits = advantage_logits - jnp.mean(advantage_logits, axis=-2, keepdims=True)
        return value_logits + centered_advantage_logits


def build_rainbow_zoo_atari_rmsprop(config: RainbowConfig) -> optax.GradientTransformation:
    """Build the rmsprop transform with a swept-value slot for its scalar hyperparameters.

    ``learning_rate``, ``decay`` and ``eps`` are injected via
    ``optax.inject_hyperparams`` so a sweep can overwrite them in ``opt_state``
    each step (see ``make_train_step``'s ``_do_learn``) instead of leaking one
    arm's value into every arm's compiled transform. ``centered`` stays a
    ``static_args`` entry: it selects a code path, not a scalar.
    """
    return optax.inject_hyperparams(optax.rmsprop, static_args=("centered",))(
        learning_rate=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        decay=jnp.asarray(config.RMSPROP_DECAY, jnp.float32),
        eps=jnp.asarray(config.OPTIMIZER_EPSILON, jnp.float32),
        centered=config.RMSPROP_CENTERED,
    )


def rainbow_zoo_atari_frames_to_env_steps(frames: int, num_action_repeats: int) -> int:
    if frames < 0:
        raise ValueError("frames must be non-negative.")
    if num_action_repeats <= 0:
        raise ValueError("num_action_repeats must be positive.")
    if frames % num_action_repeats != 0:
        raise ValueError(
            "DQN Zoo frame-counted periods must divide evenly by num_action_repeats when converting to Atari env-step semantics."
        )
    return frames // num_action_repeats


def rainbow_atari_runtime_from_dqn_zoo(
    config: RainbowConfig,
    *,
    num_iterations: int = 200,
    num_train_frames_per_iteration: int = 1_000_000,
    seed: int = 42,
) -> RainbowRuntimeConfig:
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative.")
    if num_train_frames_per_iteration < 0:
        raise ValueError("num_train_frames_per_iteration must be non-negative.")

    total_train_frames = num_iterations * num_train_frames_per_iteration
    return RainbowRuntimeConfig(
        TOTAL_TRAIN_ENV_STEPS=rainbow_zoo_atari_frames_to_env_steps(total_train_frames, config.NUM_ACTION_REPEATS),
        SEED=seed,
    )


def rainbow_zoo_atari_min_replay_capacity(config: RainbowConfig) -> int:
    return int(config.REPLAY_CAPACITY * config.MIN_REPLAY_CAPACITY_FRACTION)


def rainbow_zoo_atari_learn_period_env_steps(config: RainbowConfig) -> int:
    return rainbow_zoo_atari_frames_to_env_steps(config.LEARN_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def rainbow_zoo_atari_target_update_period_env_steps(config: RainbowConfig) -> int:
    return rainbow_zoo_atari_frames_to_env_steps(config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def rainbow_zoo_atari_should_learn(env_step: int, replay_size: int, config: RainbowConfig) -> bool:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")
    if replay_size < 0:
        raise ValueError("replay_size must be non-negative.")
    if replay_size < rainbow_zoo_atari_min_replay_capacity(config):
        return False
    return env_step % rainbow_zoo_atari_learn_period_env_steps(config) == 0


def rainbow_zoo_atari_should_update_target(env_step: int, config: RainbowConfig) -> bool:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")
    return env_step % rainbow_zoo_atari_target_update_period_env_steps(config) == 0


def _rainbow_zoo_atari_should_learn_array(
    env_step: jax.Array,
    replay_size: jax.Array,
    config: RainbowConfig,
) -> jax.Array:
    min_replay_capacity = jnp.asarray(rainbow_zoo_atari_min_replay_capacity(config), dtype=replay_size.dtype)
    learn_period_env_steps = jnp.asarray(rainbow_zoo_atari_learn_period_env_steps(config), dtype=env_step.dtype)
    return (replay_size >= min_replay_capacity) & (env_step % learn_period_env_steps == 0)


def _rainbow_zoo_atari_should_update_target_array(env_step: jax.Array, config: RainbowConfig) -> jax.Array:
    target_update_period_env_steps = jnp.asarray(rainbow_zoo_atari_target_update_period_env_steps(config), dtype=env_step.dtype)
    return env_step % target_update_period_env_steps == 0


def rainbow_zoo_atari_total_train_env_steps(runtime_config: RainbowRuntimeConfig) -> int:
    if runtime_config.TOTAL_TRAIN_ENV_STEPS < 0:
        raise ValueError("TOTAL_TRAIN_ENV_STEPS must be non-negative.")
    return runtime_config.TOTAL_TRAIN_ENV_STEPS


def rainbow_support(config: RainbowConfig) -> jax.Array:
    return jnp.linspace(config.V_MIN, config.V_MAX, config.NUM_ATOMS, dtype=jnp.float32)


def _rainbow_support_from_hypers(hypers: "RainbowHypers", num_atoms: int) -> jax.Array:
    """Rebuild the support array from traced endpoints so a sweep can vary ``V_MIN``/``V_MAX``.

    ``num_atoms`` stays the static array length (``RainbowConfig.NUM_ATOMS``);
    only the endpoints come from the traced ``hypers``.
    """
    return jnp.linspace(hypers.V_MIN, hypers.V_MAX, num_atoms, dtype=jnp.float32)


def rainbow_probabilities(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


def rainbow_expected_q_values(logits: jax.Array, support: jax.Array) -> jax.Array:
    return categorical_expected_value(rainbow_probabilities(logits), support)


def rainbow_select_actions(logits: jax.Array, support: jax.Array) -> jax.Array:
    return jnp.argmax(rainbow_expected_q_values(logits, support), axis=-1)


def categorical_target_support(
    rewards: jax.Array,
    support: jax.Array,
    discounts: jax.Array,
) -> jax.Array:
    """Shift and scale the atom support into the target distribution's support.

    ``discounts`` is the per-transition bootstrap coefficient stored in replay, which is
    ``prod_{j<m} d`` over the n-step window that transition actually spanned. It carries
    termination on its own -- a terminal window ends in ``d = 0`` and the whole product
    collapses -- so there is no separate ``done`` mask to apply, and a truncated window
    keeps the shortened bootstrap a mask would have destroyed.

    Args:
        rewards: The transitions' accumulated n-step rewards, shaped ``(batch,)``.
        support: The fixed atom locations, shaped ``(num_atoms,)``.
        discounts: Per-transition bootstrap coefficients, broadcastable against
            ``rewards``.

    Returns:
        The target support, shaped ``(batch, num_atoms)``.
    """
    rewards = jnp.asarray(rewards, dtype=support.dtype)
    discounts = jnp.asarray(discounts, dtype=support.dtype)
    return rewards[..., None] + discounts[..., None] * support


def categorical_target_probabilities(
    rewards: jax.Array,
    next_probabilities: jax.Array,
    support: jax.Array,
    discounts: jax.Array,
) -> jax.Array:
    target_support = categorical_target_support(rewards, support, discounts)
    return categorical_l2_project(target_support, next_probabilities, support)


def categorical_loss(logits: jax.Array, target_probabilities: jax.Array) -> jax.Array:
    return jnp.mean(categorical_cross_entropy(logits, target_probabilities))


def categorical_losses(logits: jax.Array, target_probabilities: jax.Array) -> jax.Array:
    return categorical_cross_entropy(logits, target_probabilities)


def _gather_action_logits(action_logits: jax.Array, actions: jax.Array) -> jax.Array:
    return jnp.take_along_axis(action_logits, actions[..., None, None], axis=-2).squeeze(-2)


def rainbow_loss(
    params: VariableDict,
    target_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    importance_weights: jax.Array,
    key: jax.Array,
    *,
    apply_fn: RainbowApplyFn,
    support: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Importance-weighted cross-entropy against the double-Q distributional target.

    Reachable from outside the agent so the termination semantics can be gated on the
    real target rule rather than on a copy of it. ``discounts`` carries the cumulative
    n-step coefficient replay stored, so a row whose window ended in a termination
    collapses onto its bare reward and the loss becomes independent of ``next_obs``
    there, while a row cut short by a truncation keeps the shorter bootstrap it earned.

    Args:
        params: Online parameters. Differentiate with respect to this argument only.
        target_params: Target-network parameters, which value the bootstrap action the
            online network selects.
        obs: Observations the stored transitions started from.
        actions: Actions taken from ``obs``, as integer indices.
        rewards: Accumulated n-step rewards of the stored transitions.
        next_obs: True observations the stored windows reached.
        discounts: Per-transition bootstrap coefficients, ``0.0`` on a terminated window.
        importance_weights: Prioritised-replay weights, held out of the gradient.
        key: PRNG key for the three independent noisy-network samples this loss draws.
        apply_fn: The network's ``apply``, static because it is a Python callable.
        support: The atom locations, which are also the target's projection basis.

    Returns:
        The scalar weighted loss and the per-example losses the priorities are set from.
    """
    obs_noise_key, next_online_noise_key, next_target_noise_key = jax.random.split(key, 3)

    logits = apply_fn(params, obs, rngs={"noise": obs_noise_key})
    chosen_logits = _gather_action_logits(logits, actions)

    next_online_logits = apply_fn(params, next_obs, rngs={"noise": next_online_noise_key})
    next_actions = rainbow_select_actions(next_online_logits, support)
    next_target_logits = apply_fn(target_params, next_obs, rngs={"noise": next_target_noise_key})
    next_target_probabilities = rainbow_probabilities(_gather_action_logits(next_target_logits, next_actions))

    target_probabilities = categorical_target_probabilities(rewards, next_target_probabilities, support, discounts)
    per_example_loss = categorical_losses(chosen_logits, jax.lax.stop_gradient(target_probabilities))
    weighted_loss = per_example_loss * jax.lax.stop_gradient(importance_weights)
    return jnp.mean(weighted_loss), per_example_loss


def _init_n_step_accumulator(config: RainbowConfig, prototype: RainbowTransition) -> _NStepAccumulatorState:
    if config.N_STEP <= 0:
        raise ValueError("N_STEP must be positive.")
    return _NStepAccumulatorState(
        obs=jnp.zeros((config.N_STEP, *prototype.obs.shape), dtype=prototype.obs.dtype),
        action=jnp.zeros((config.N_STEP, *prototype.action.shape), dtype=prototype.action.dtype),
        reward=jnp.zeros((config.N_STEP, *prototype.reward.shape), dtype=prototype.reward.dtype),
        discount=jnp.zeros((config.N_STEP, *prototype.discount.shape), dtype=prototype.discount.dtype),
        size=jnp.zeros((), dtype=jnp.int32),
    )


def _accumulate_n_step_step(
    accumulator: _NStepAccumulatorState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    discount: jax.Array,
) -> _NStepAccumulatorState:
    """Open a pending transition at this step and fold the step into every pending one.

    The fold is the recursion behind the n-step target: a row that has already consumed
    ``m`` steps holds ``reward = sum_{k<m} (prod_{j<k} d) r`` and ``discount = prod_{j<m} d``,
    so consuming one more step is ``reward += discount * r`` followed by ``discount *= d``.
    Applying it to the row opened at this step -- primed with ``reward = 0`` and
    ``discount = 1`` -- gives that row its own first term with no special case.
    """
    slot = accumulator.size.astype(jnp.int32)
    size = accumulator.size + jnp.asarray(1, dtype=accumulator.size.dtype)
    pending = jnp.arange(accumulator.reward.shape[0], dtype=jnp.int32) < size
    opened_reward = accumulator.reward.at[slot].set(jnp.zeros_like(reward))
    opened_discount = accumulator.discount.at[slot].set(jnp.ones_like(discount))
    return _NStepAccumulatorState(
        obs=accumulator.obs.at[slot].set(obs),
        action=accumulator.action.at[slot].set(action),
        reward=jnp.where(pending, opened_reward + opened_discount * reward, opened_reward),
        discount=jnp.where(pending, opened_discount * discount, opened_discount),
        size=size,
    )


def _shift_n_step_accumulator_left(accumulator: _NStepAccumulatorState) -> _NStepAccumulatorState:
    return _NStepAccumulatorState(
        obs=jnp.concatenate([accumulator.obs[1:], jnp.zeros_like(accumulator.obs[:1])], axis=0),
        action=jnp.concatenate([accumulator.action[1:], jnp.zeros_like(accumulator.action[:1])], axis=0),
        reward=jnp.concatenate([accumulator.reward[1:], jnp.zeros_like(accumulator.reward[:1])], axis=0),
        discount=jnp.concatenate([accumulator.discount[1:], jnp.zeros_like(accumulator.discount[:1])], axis=0),
        size=accumulator.size - jnp.asarray(1, dtype=accumulator.size.dtype),
    )


def _materialize_n_step_transitions(
    config: RainbowConfig,
    accumulator: _NStepAccumulatorState,
    next_obs: jax.Array,
) -> RainbowTransition:
    """Read the pending window out as replay entries, one per row.

    ``next_obs`` is the bootstrap observation this step reached, and it is the right
    bootstrap state for *every* row: a flush happens at one step, so the rows leave with
    different horizons ``m_t`` but with the same ``t + m_t``.
    """
    return RainbowTransition(
        obs=accumulator.obs,
        action=accumulator.action,
        reward=accumulator.reward,
        next_obs=jnp.broadcast_to(next_obs, (config.N_STEP, *next_obs.shape)),
        discount=accumulator.discount,
    )


def _advance_n_step_accumulator(
    config: RainbowConfig,
    accumulator: _NStepAccumulatorState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_obs: jax.Array,
    discount: jax.Array,
    episode_end: jax.Array,
) -> tuple[_NStepAccumulatorState, RainbowTransition, jax.Array]:
    """Fold one step in, then report which pending rows this step completes.

    Two separate boundary quantities arrive per step, and they disagree exactly on a
    truncation. ``discount`` is the bootstrap coefficient, which is ``0`` only on
    termination and therefore erases the tail of the return by itself. ``episode_end``
    is the trajectory break, which is true on truncation too, and it is the only thing
    that cuts a window whose discount is still nonzero.

    Args:
        config: Static hyperparameters; only ``N_STEP`` is read.
        accumulator: The pending window before this step.
        obs: Observation the transition this step opens started from.
        action: Action the transition this step opens took.
        reward: Reward this step earned.
        next_obs: True observation this step reached, never a post-reset one.
        discount: Bootstrap coefficient for this step, ``0`` on termination.
        episode_end: Whether this step broke the trajectory.

    Returns:
        The window after this step, the replay entries for every row of the window as it
        stood *before* the flush, and the mask selecting the rows to insert.
    """
    discount = jnp.asarray(discount, dtype=accumulator.discount.dtype)
    appended = _accumulate_n_step_step(accumulator, obs, action, reward, discount)
    transitions = _materialize_n_step_transitions(config, appended, next_obs)

    indices = jnp.arange(config.N_STEP, dtype=jnp.int32)
    appended_is_full = appended.size == jnp.asarray(config.N_STEP, dtype=appended.size.dtype)
    insert_mask = jnp.where(episode_end, indices < appended.size, jnp.logical_and(appended_is_full, indices == 0))

    next_accumulator = jax.lax.cond(
        episode_end,
        lambda value: jax.tree_util.tree_map(jnp.zeros_like, value),
        lambda value: jax.lax.cond(appended_is_full, _shift_n_step_accumulator_left, lambda inner: inner, value),
        appended,
    )
    return next_accumulator, transitions, insert_mask


def _per_add_batched_transitions(
    buffer_state: PERBufferState,
    transitions: RainbowTransition,
    insert_mask: jax.Array,
) -> PERBufferState:
    def _add_one(state: PERBufferState, xs: tuple[RainbowTransition, jax.Array]) -> tuple[PERBufferState, None]:
        transition, should_add = xs
        state = jax.lax.cond(
            should_add,
            lambda args: per_add(args[0], args[1]),
            lambda args: args[0],
            (state, transition),
        )
        return state, None

    buffer_state, _ = jax.lax.scan(_add_one, buffer_state, (transitions, insert_mask))
    return buffer_state


def _replay_prototype_from_state(buffer_state: PERBufferState) -> RainbowTransition:
    """Rebuild the entry shape ``per_sample`` reassembles a batch into.

    ``jax_replay.per`` stores its leaves keyed by stringified flatten index and needs a
    pytree of that shape to unflatten a sampled batch. Every leaf's shape and dtype is
    already recorded in the stored arrays, so ``step`` needs nothing static in its carry
    to reach its own buffer.
    """
    return RainbowTransition(
        **{name: buffer_state.data[str(index)][0] for index, name in enumerate(RainbowTransition._fields)}
    )


@chex_struct(frozen=True)
class RainbowAgentState:
    """Everything the Rainbow agent carries between loop iterations.

    Attributes:
        train_state: Online parameters, optimizer state, and the static ``apply_fn``
            through which ``step`` reaches the noisy duelling C51 network.
        target_params: Parameters of the target network the bootstrap distribution is
            read from.
        buffer_state: Prioritised replay contents, which are also the only record of the
            entry's pytree shape.
        n_step: The window of transitions whose n-step return is still incomplete. It is
            the one piece of agent state the boundary semantics are visible in.
        last_obs: Observation the pending transition started from. Zero-primed at
            ``init``; never read before the first insertion, which is guarded on
            ``step_index > 0``.
        last_action: Action that opened the pending transition, zero-primed likewise.
        key: PRNG key for sampling minibatches and for the network's noise.
        hypers: Swept hyperparameters, traced so a batch of arms shares one kernel.
    """

    train_state: TrainState
    target_params: VariableDict
    buffer_state: PERBufferState
    n_step: _NStepAccumulatorState
    last_obs: jax.Array
    last_action: jax.Array
    key: chex.PRNGKey
    hypers: RainbowHypers


class RainbowAgent:
    """Rainbow: distributional double-Q learning over prioritised n-step replay."""

    def __init__(self, config: RainbowConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> RainbowAgentState:
        """Build parameters, optimizer, replay, the n-step window and the pending slots.

        Args:
            key: PRNG key for parameter and noise initialization.
            spec: The environment description. Must be discrete -- the target maximizes
                an expected value over the action space -- and must carry image
                observations the Nature CNN can consume.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space.
        """
        if spec.num_actions is None:
            raise ValueError(f"Rainbow requires a discrete action space, got spec {spec.id!r} with none")

        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_shape = tuple(spec.action_shape)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = RainbowNatureNetwork(
            action_dim=spec.num_actions,
            num_atoms=self.config.NUM_ATOMS,
            observation_layout=infer_nature_observation_layout(observation_shape),
        )

        params_key, noise_key, carry_key = jax.random.split(key, 3)
        params = network.init(
            {"params": params_key, "noise": noise_key},
            jnp.zeros(observation_shape, dtype=observation_dtype),
        )
        replay_prototype = RainbowTransition(
            obs=jnp.zeros(observation_shape, dtype=observation_dtype),
            action=jnp.zeros(action_shape, dtype=action_dtype),
            reward=jnp.zeros((), dtype=jnp.float32),
            next_obs=jnp.zeros(observation_shape, dtype=observation_dtype),
            discount=jnp.zeros((), dtype=jnp.float32),
        )

        return RainbowAgentState(
            train_state=TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=build_rainbow_zoo_atari_rmsprop(self.config),
            ),
            target_params=params,
            buffer_state=init_per_buffer(replay_prototype, self.config.REPLAY_CAPACITY),
            n_step=_init_n_step_accumulator(self.config, replay_prototype),
            last_obs=jnp.zeros(observation_shape, dtype=observation_dtype),
            last_action=jnp.zeros(action_shape, dtype=action_dtype),
            key=carry_key,
            hypers=rainbow_hypers(self.config),
        )

    def step(
        self,
        state: RainbowAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[RainbowAgentState, jax.Array]:
        """Fold the pending transition into the window, learn, sync the target, then act.

        The order is normative. The fold uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and the whole window would bootstrap from the next episode's first
        state -- for an n-step agent that corrupts up to ``N_STEP`` entries at once, not
        the one a single-step agent would lose.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so the
                fold is guarded above it, and every schedule keys off it.

        Returns:
            The next state, the action to apply, and a fixed two-key metric schema.
            ``loss`` is a zero placeholder on the iterations that do not learn, so the
            pytree returned to ``lax.scan`` is identical on every iteration.
        """
        hypers = state.hypers
        support = _rainbow_support_from_hypers(hypers, self.config.NUM_ATOMS)
        carry_key, sample_key, learn_key, action_noise_key = jax.random.split(state.key, 4)

        n_step, buffer_state = jax.lax.cond(
            step_index > 0,
            lambda: self._close_pending_transition(state, timestep),
            lambda: (state.n_step, state.buffer_state),
        )

        can_learn = _rainbow_zoo_atari_should_learn_array(step_index, buffer_state.count, self.config)
        train_state, buffer_state, loss = jax.lax.cond(
            can_learn,
            lambda: self._learn(
                state.train_state,
                state.target_params,
                buffer_state,
                sample_key,
                learn_key,
                hypers,
                support,
            ),
            lambda: (state.train_state, buffer_state, jnp.zeros((), jnp.float32)),
        )

        target_params = jax.lax.cond(
            _rainbow_zoo_atari_should_update_target_array(step_index, self.config),
            lambda: train_state.params,
            lambda: state.target_params,
        )

        action_logits = jnp.asarray(
            train_state.apply_fn(train_state.params, timestep.observation, rngs={"noise": action_noise_key})
        )
        action = rainbow_select_actions(action_logits, support).astype(state.last_action.dtype)

        return AgentStep(
            state=RainbowAgentState(
                train_state=train_state,
                target_params=target_params,
                buffer_state=buffer_state,
                n_step=n_step,
                last_obs=timestep.observation,
                last_action=action,
                key=carry_key,
                hypers=hypers,
            ),
            action=action,
            metrics={
                "loss": loss,
                "max_q": jnp.max(rainbow_expected_q_values(action_logits, support)).astype(jnp.float32),
            },
        )

    def _close_pending_transition(
        self,
        state: RainbowAgentState,
        timestep: Timestep[jax.Array],
    ) -> tuple[_NStepAccumulatorState, PERBufferState]:
        """Fold the closed transition into the window and insert whatever it completed.

        The window completes nothing on most steps, one entry once it is full, and every
        pending entry at a boundary, so the insertion is masked rather than branched.
        """
        n_step, transitions, insert_mask = _advance_n_step_accumulator(
            self.config,
            state.n_step,
            state.last_obs,
            state.last_action,
            timestep.reward,
            timestep.bootstrap_observation,
            timestep.discount,
            timestep.episode_end,
        )
        return n_step, _per_add_batched_transitions(state.buffer_state, transitions, insert_mask)

    def _learn(
        self,
        train_state: TrainState,
        target_params: VariableDict,
        buffer_state: PERBufferState,
        sample_key: chex.PRNGKey,
        noise_key: chex.PRNGKey,
        hypers: RainbowHypers,
        support: jax.Array,
    ) -> tuple[TrainState, PERBufferState, jax.Array]:
        """Take one gradient step on a prioritised minibatch, then reprioritise it.

        ``LEARNING_RATE``/``RMSPROP_DECAY``/``OPTIMIZER_EPSILON`` are traced hypers rather
        than the static config the optimizer was constructed with, so each arm's current
        value is pushed into the injected optimizer state before every gradient step --
        otherwise every vmapped arm would silently train at whichever value the transform
        was built with.
        """
        sampled_batch, importance_weights, batch_indices = per_sample(
            buffer_state,
            sample_key,
            batch_size=self.config.BATCH_SIZE,
            beta=1.0,
            prototype=_replay_prototype_from_state(buffer_state),
        )

        opt_state = train_state.opt_state
        train_state = train_state.replace(
            opt_state=opt_state._replace(
                hyperparams={
                    **opt_state.hyperparams,
                    "learning_rate": hypers.LEARNING_RATE,
                    "decay": hypers.RMSPROP_DECAY,
                    "eps": hypers.OPTIMIZER_EPSILON,
                }
            )
        )

        def _loss_with_priorities(params: VariableDict) -> tuple[jax.Array, jax.Array]:
            return rainbow_loss(
                params,
                target_params,
                sampled_batch.obs,
                sampled_batch.action,
                sampled_batch.reward,
                sampled_batch.next_obs,
                sampled_batch.discount,
                importance_weights,
                noise_key,
                apply_fn=train_state.apply_fn,
                support=support,
            )

        (loss, per_example_loss), grads = jax.value_and_grad(_loss_with_priorities, has_aux=True)(train_state.params)
        buffer_state = per_update_priorities(buffer_state, batch_indices, per_example_loss)
        return train_state.apply_gradients(grads=grads), buffer_state, loss.astype(jnp.float32)
