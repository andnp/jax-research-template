from typing import Callable, Literal, NamedTuple, TypedDict

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
from jax_nn.layers import NatureCNN, NoisyLinear
from jax_nn.typed_module import TypedApply
from jax_replay.per import init_per_buffer, per_add, per_sample, per_update_priorities
from jax_replay.types import PERBufferState
from rl_components.gym_env import DiscreteActionSpace, GymEnv
from rl_components.structs import chex_struct

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
    ADDITIONAL_DISCOUNT: float = 0.99
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
    """The hyperparameters ``train`` reads as traced values.

    These ride as a ``train(rng, hypers)`` argument rather than the closed-over
    ``RainbowConfig`` so a sweep can ``jax.vmap`` one compiled kernel across a
    batch of arms. Everything here is read only in arithmetic, never to size an
    array or take a Python branch -- those stay on ``RainbowConfig``.
    """

    LEARNING_RATE: jax.Array
    OPTIMIZER_EPSILON: jax.Array
    RMSPROP_DECAY: jax.Array
    V_MIN: jax.Array
    V_MAX: jax.Array
    ADDITIONAL_DISCOUNT: jax.Array


def rainbow_hypers(config: RainbowConfig) -> RainbowHypers:
    return RainbowHypers(
        LEARNING_RATE=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        OPTIMIZER_EPSILON=jnp.asarray(config.OPTIMIZER_EPSILON, jnp.float32),
        RMSPROP_DECAY=jnp.asarray(config.RMSPROP_DECAY, jnp.float32),
        V_MIN=jnp.asarray(config.V_MIN, jnp.float32),
        V_MAX=jnp.asarray(config.V_MAX, jnp.float32),
        ADDITIONAL_DISCOUNT=jnp.asarray(config.ADDITIONAL_DISCOUNT, jnp.float32),
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
        discount: Bootstrap coefficient for this step, ``0`` on termination.
        next_obs: True observation this step reached, never a post-reset one.
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


def initialize_train_state(
    config: RainbowConfig,
    env: GymEnv[DiscreteActionSpace],
    rng: jax.Array,
    env_params: object | None = None,
) -> tuple[RainbowNatureNetwork, RainbowTransition, "RunnerState"]:
    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    observation_shape = tuple(observation_space.shape)
    network = RainbowNatureNetwork(
        action_dim=action_space.n,
        num_atoms=config.NUM_ATOMS,
        observation_layout=infer_nature_observation_layout(observation_shape),
    )

    rng, init_rng, init_noise_rng = jax.random.split(rng, 3)
    init_x = jnp.zeros(observation_shape, dtype=observation_space.dtype)
    params = network.init({"params": init_rng, "noise": init_noise_rng}, init_x)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=build_rainbow_zoo_atari_rmsprop(config),
    )
    target_params = params

    replay_prototype = RainbowTransition(
        obs=jnp.zeros(observation_shape, dtype=observation_space.dtype),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_obs=jnp.zeros(observation_shape, dtype=observation_space.dtype),
        discount=jnp.zeros((), dtype=jnp.float32),
    )
    buffer_state = init_per_buffer(replay_prototype, config.REPLAY_CAPACITY)
    n_step_accumulator = _init_n_step_accumulator(config, replay_prototype)

    rng, reset_rng = jax.random.split(rng)
    last_obs, env_state = env.reset(reset_rng, env_params)
    runner_state = RunnerState(
        train_state=train_state,
        target_params=target_params,
        buffer_state=buffer_state,
        env_state=env_state,
        last_obs=last_obs,
        rng=rng,
        n_step_accumulator=n_step_accumulator,
    )
    return network, replay_prototype, runner_state


def make_train_step(
    config: RainbowConfig,
    runtime_config: RainbowRuntimeConfig,
    env: GymEnv[DiscreteActionSpace],
    network: RainbowNatureNetwork,
    replay_prototype: RainbowTransition,
    hypers: RainbowHypers,
    env_params: object | None = None,
) -> Callable[["RunnerState", jax.Array], tuple["RunnerState", dict[str, jax.Array]]]:
    del runtime_config
    # Derived from the traced `hypers` (not the static `config`) so a
    # `jax.vmap` sweep over `V_MIN`/`V_MAX` recomputes the support per arm
    # instead of baking one arm's value in here.
    support = _rainbow_support_from_hypers(hypers, config.NUM_ATOMS)

    def train_step(runner_state: "RunnerState", step_index: jax.Array) -> tuple["RunnerState", dict[str, jax.Array]]:
        train_state, target_params, buffer_state, env_state, last_obs, rng, n_step_accumulator = runner_state

        env_step = step_index + 1
        rng, action_noise_rng, step_rng, sample_rng, learn_rng = jax.random.split(rng, 5)
        action_logits = network.apply(train_state.params, last_obs, rngs={"noise": action_noise_rng})
        action = rainbow_select_actions(action_logits, support)

        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
        n_step_accumulator, finalized_transitions, insert_mask = _advance_n_step_accumulator(
            config,
            n_step_accumulator,
            last_obs,
            action,
            reward,
            obs,
            jnp.where(done, jnp.zeros((), jnp.float32), hypers.ADDITIONAL_DISCOUNT),
            done,
        )
        buffer_state = _per_add_batched_transitions(buffer_state, finalized_transitions, insert_mask)

        can_learn = _rainbow_zoo_atari_should_learn_array(env_step, buffer_state.count, config)

        def _do_learn(args: tuple[TrainState, VariableDict, PERBufferState, jax.Array]) -> tuple[TrainState, PERBufferState, jax.Array]:
            train_state, target_params, buffer_state, learn_rng = args
            sampled_batch, importance_weights, batch_indices = per_sample(
                buffer_state,
                sample_rng,
                batch_size=config.BATCH_SIZE,
                beta=1.0,
                prototype=replay_prototype,
            )
            # `LEARNING_RATE`/`RMSPROP_DECAY`/`OPTIMIZER_EPSILON` are traced
            # `hypers`, not the static config the optimizer was constructed
            # with, so each arm's current value must be pushed into the
            # injected optimizer state before every gradient step -- otherwise
            # every vmapped arm would silently train at whichever value the
            # transform was built with.
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
                    learn_rng,
                    apply_fn=network.apply,
                    support=support,
                )

            (loss, per_example_loss), grads = jax.value_and_grad(_loss_with_priorities, has_aux=True)(
                train_state.params,
            )
            buffer_state = per_update_priorities(buffer_state, batch_indices, per_example_loss)
            return train_state.apply_gradients(grads=grads), buffer_state, loss

        def _skip_learn(args: tuple[TrainState, VariableDict, PERBufferState, jax.Array]) -> tuple[TrainState, PERBufferState, jax.Array]:
            train_state, _target_params, buffer_state, _learn_rng = args
            return train_state, buffer_state, jnp.asarray(0.0)

        train_state, buffer_state, loss = jax.lax.cond(
            can_learn,
            _do_learn,
            _skip_learn,
            (train_state, target_params, buffer_state, learn_rng),
        )

        target_params = jax.lax.cond(
            _rainbow_zoo_atari_should_update_target_array(env_step, config),
            lambda: train_state.params,
            lambda: target_params,
        )

        step_metrics = dict(info)
        step_metrics["loss"] = loss
        step_metrics["max_q"] = jnp.max(rainbow_expected_q_values(action_logits, support))

        next_runner_state = RunnerState(
            train_state=train_state,
            target_params=target_params,
            buffer_state=buffer_state,
            env_state=env_state,
            last_obs=obs,
            rng=rng,
            n_step_accumulator=n_step_accumulator,
        )
        return next_runner_state, step_metrics

    return train_step


class RunnerState(NamedTuple):
    train_state: TrainState
    target_params: VariableDict
    buffer_state: PERBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array
    n_step_accumulator: _NStepAccumulatorState


class RainbowTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(
    config: RainbowConfig,
    runtime_config: RainbowRuntimeConfig,
    env: GymEnv[DiscreteActionSpace],
    env_params: object | None = None,
) -> Callable[[jax.Array, RainbowHypers | None], RainbowTrainOutput]:
    total_env_steps = rainbow_zoo_atari_total_train_env_steps(runtime_config)

    def train(rng: jax.Array, hypers: RainbowHypers | None = None) -> RainbowTrainOutput:
        if hypers is None:
            hypers = rainbow_hypers(config)
        network, replay_prototype, runner_state = initialize_train_state(config, env, rng, env_params)
        train_step = make_train_step(config, runtime_config, env, network, replay_prototype, hypers, env_params)
        runner_state, metrics = jax.lax.scan(train_step, runner_state, jnp.arange(total_env_steps))
        return {"runner_state": runner_state, "metrics": metrics}

    return train