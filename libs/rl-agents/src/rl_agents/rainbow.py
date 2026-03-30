from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax_nn.distributional import (
    categorical_cross_entropy,
    categorical_expected_value,
    categorical_l2_project,
)
from jax_nn.layers import NatureCNN, NoisyLinear
from jax_replay.per import init_per_buffer, per_add, per_sample, per_update_priorities
from jax_replay.types import PERBufferState
from rl_components.structs import chex_struct

from rl_agents.dqn import _EnvLike, _infer_nature_observation_layout, _prepare_nature_observations


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


class RainbowTransition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array


@chex_struct(frozen=True, kw_only=True)
class _NStepAccumulatorState:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    size: jax.Array


class RainbowNatureNetwork(nn.Module):
    action_dim: int
    num_atoms: int
    observation_layout: Literal["hwc", "fhwc"]
    dtype: jnp.dtype = jnp.float32

    if TYPE_CHECKING:
        def apply(
            self,
            variables: object,
            x: jax.Array,
            *,
            rngs: object | None = None,
        ) -> jax.Array: ...

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = _prepare_nature_observations(x, self.observation_layout)
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
    return optax.rmsprop(
        learning_rate=config.LEARNING_RATE,
        decay=config.RMSPROP_DECAY,
        eps=config.OPTIMIZER_EPSILON,
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
):
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


def rainbow_zoo_atari_total_train_env_steps(runtime_config: RainbowRuntimeConfig) -> int:
    if runtime_config.TOTAL_TRAIN_ENV_STEPS < 0:
        raise ValueError("TOTAL_TRAIN_ENV_STEPS must be non-negative.")
    return runtime_config.TOTAL_TRAIN_ENV_STEPS


def rainbow_support(config: RainbowConfig) -> jax.Array:
    return jnp.linspace(config.V_MIN, config.V_MAX, config.NUM_ATOMS, dtype=jnp.float32)


def rainbow_probabilities(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


def rainbow_expected_q_values(logits: jax.Array, support: jax.Array) -> jax.Array:
    return categorical_expected_value(rainbow_probabilities(logits), support)


def rainbow_select_actions(logits: jax.Array, support: jax.Array) -> jax.Array:
    return jnp.argmax(rainbow_expected_q_values(logits, support), axis=-1)


def categorical_target_support(
    rewards: jax.Array,
    dones: jax.Array,
    support: jax.Array,
    discount: float,
) -> jax.Array:
    rewards = jnp.asarray(rewards, dtype=support.dtype)
    dones = jnp.asarray(dones, dtype=support.dtype)
    return rewards[..., None] + discount * support * (1.0 - dones[..., None])


def categorical_target_probabilities(
    rewards: jax.Array,
    dones: jax.Array,
    next_probabilities: jax.Array,
    support: jax.Array,
    discount: float,
) -> jax.Array:
    target_support = categorical_target_support(rewards, dones, support, discount)
    return categorical_l2_project(target_support, next_probabilities, support)


def categorical_loss(logits: jax.Array, target_probabilities: jax.Array) -> jax.Array:
    return jnp.mean(categorical_cross_entropy(logits, target_probabilities))


def categorical_losses(logits: jax.Array, target_probabilities: jax.Array) -> jax.Array:
    return categorical_cross_entropy(logits, target_probabilities)


def _gather_action_logits(action_logits: jax.Array, actions: jax.Array) -> jax.Array:
    return jnp.take_along_axis(action_logits, actions[..., None, None], axis=-2).squeeze(-2)


def _init_n_step_accumulator(config: RainbowConfig, prototype: RainbowTransition) -> _NStepAccumulatorState:
    if config.N_STEP <= 0:
        raise ValueError("N_STEP must be positive.")
    return _NStepAccumulatorState(
        obs=jnp.zeros((config.N_STEP, *prototype.obs.shape), dtype=prototype.obs.dtype),
        action=jnp.zeros((config.N_STEP, *prototype.action.shape), dtype=prototype.action.dtype),
        reward=jnp.zeros((config.N_STEP, *prototype.reward.shape), dtype=prototype.reward.dtype),
        size=jnp.zeros((), dtype=jnp.int32),
    )


def _append_n_step_raw_transition(
    accumulator: _NStepAccumulatorState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
) -> _NStepAccumulatorState:
    insert_index = accumulator.size.astype(jnp.int32)
    return _NStepAccumulatorState(
        obs=accumulator.obs.at[insert_index].set(obs),
        action=accumulator.action.at[insert_index].set(action),
        reward=accumulator.reward.at[insert_index].set(reward),
        size=accumulator.size + jnp.asarray(1, dtype=accumulator.size.dtype),
    )


def _shift_n_step_accumulator_left(accumulator: _NStepAccumulatorState) -> _NStepAccumulatorState:
    return _NStepAccumulatorState(
        obs=jnp.concatenate([accumulator.obs[1:], jnp.zeros_like(accumulator.obs[:1])], axis=0),
        action=jnp.concatenate([accumulator.action[1:], jnp.zeros_like(accumulator.action[:1])], axis=0),
        reward=jnp.concatenate([accumulator.reward[1:], jnp.zeros_like(accumulator.reward[:1])], axis=0),
        size=accumulator.size - jnp.asarray(1, dtype=accumulator.size.dtype),
    )


def _materialize_n_step_transitions(
    config: RainbowConfig,
    prototype: RainbowTransition,
    accumulator: _NStepAccumulatorState,
    next_obs: jax.Array,
    done: jax.Array,
) -> RainbowTransition:
    indices = jnp.arange(config.N_STEP, dtype=jnp.int32)
    reward_offsets = indices[None, :] - indices[:, None]
    future_reward_mask = reward_offsets >= 0
    discount = jnp.asarray(config.ADDITIONAL_DISCOUNT, dtype=prototype.reward.dtype)
    discount_matrix = jnp.where(
        future_reward_mask,
        discount ** reward_offsets.astype(prototype.reward.dtype),
        jnp.zeros((config.N_STEP, config.N_STEP), dtype=prototype.reward.dtype),
    )
    valid_rows = indices[:, None] < accumulator.size
    valid_columns = indices[None, :] < accumulator.size
    reward_weights = discount_matrix * jnp.asarray(
        future_reward_mask & valid_rows & valid_columns,
        dtype=prototype.reward.dtype,
    )
    rewards = reward_weights @ accumulator.reward
    return RainbowTransition(
        obs=accumulator.obs,
        action=accumulator.action,
        reward=rewards.astype(prototype.reward.dtype),
        next_obs=jnp.broadcast_to(next_obs, (config.N_STEP, *prototype.next_obs.shape)),
        done=jnp.broadcast_to(jnp.asarray(done, dtype=prototype.done.dtype), (config.N_STEP,)),
    )


def _advance_n_step_accumulator(
    config: RainbowConfig,
    prototype: RainbowTransition,
    accumulator: _NStepAccumulatorState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
) -> tuple[_NStepAccumulatorState, RainbowTransition, jax.Array]:
    appended = _append_n_step_raw_transition(accumulator, obs, action, reward)
    transitions = _materialize_n_step_transitions(config, prototype, appended, next_obs, done)

    indices = jnp.arange(config.N_STEP, dtype=jnp.int32)
    appended_is_full = appended.size == jnp.asarray(config.N_STEP, dtype=appended.size.dtype)
    insert_mask = jnp.where(done, indices < appended.size, jnp.logical_and(appended_is_full, indices == 0))

    next_accumulator = jax.lax.cond(
        done,
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
    def _add_one(state, xs):
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
    env: object,
    rng: jax.Array,
    env_params: object | None = None,
):
    env = cast(_EnvLike, env)

    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    observation_shape = tuple(observation_space.shape)
    network = RainbowNatureNetwork(
        action_dim=action_space.n,
        num_atoms=config.NUM_ATOMS,
        observation_layout=_infer_nature_observation_layout(observation_shape),
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
        done=jnp.zeros((), dtype=jnp.bool_),
    )
    buffer_state = init_per_buffer(replay_prototype, config.REPLAY_CAPACITY)
    n_step_accumulator = _init_n_step_accumulator(config, replay_prototype)

    rng, reset_rng = jax.random.split(rng)
    last_obs, env_state = env.reset(reset_rng, env_params)
    runner_state = (train_state, target_params, buffer_state, env_state, last_obs, rng, n_step_accumulator)
    return network, replay_prototype, runner_state


def make_train_step(
    config: RainbowConfig,
    runtime_config: RainbowRuntimeConfig,
    env: object,
    network: RainbowNatureNetwork,
    replay_prototype: RainbowTransition,
    env_params: object | None = None,
):
    env = cast(_EnvLike, env)

    del runtime_config
    support = rainbow_support(config)
    bootstrap_discount = config.ADDITIONAL_DISCOUNT**config.N_STEP
    min_replay_capacity = rainbow_zoo_atari_min_replay_capacity(config)
    learn_period_env_steps = rainbow_zoo_atari_learn_period_env_steps(config)
    target_update_period_env_steps = rainbow_zoo_atari_target_update_period_env_steps(config)

    def _loss(
        params: object,
        target_params: object,
        obs: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_obs: jax.Array,
        dones: jax.Array,
        online_next_noise_rng: jax.Array,
        target_next_noise_rng: jax.Array,
        online_obs_noise_rng: jax.Array,
        importance_weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = network.apply(params, obs, rngs={"noise": online_obs_noise_rng})
        chosen_logits = _gather_action_logits(logits, actions)

        next_online_logits = network.apply(params, next_obs, rngs={"noise": online_next_noise_rng})
        next_actions = rainbow_select_actions(next_online_logits, support)
        next_target_logits = network.apply(target_params, next_obs, rngs={"noise": target_next_noise_rng})
        next_target_selected_logits = _gather_action_logits(next_target_logits, next_actions)
        next_target_probabilities = rainbow_probabilities(next_target_selected_logits)
        target_probabilities = categorical_target_probabilities(
            rewards,
            dones,
            next_target_probabilities,
            support,
            bootstrap_discount,
        )
        per_example_loss = categorical_losses(chosen_logits, jax.lax.stop_gradient(target_probabilities))
        weighted_loss = per_example_loss * jax.lax.stop_gradient(importance_weights)
        return jnp.mean(weighted_loss), per_example_loss

    def train_step(runner_state, step_index):
        train_state, target_params, buffer_state, env_state, last_obs, rng, n_step_accumulator = runner_state

        env_step = step_index + 1
        rng, action_noise_rng, step_rng, sample_rng, learn_rng = jax.random.split(rng, 5)
        action_logits = network.apply(train_state.params, last_obs, rngs={"noise": action_noise_rng})
        action = rainbow_select_actions(action_logits, support)

        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
        n_step_accumulator, finalized_transitions, insert_mask = _advance_n_step_accumulator(
            config,
            replay_prototype,
            n_step_accumulator,
            last_obs,
            action,
            reward,
            obs,
            done,
        )
        buffer_state = _per_add_batched_transitions(buffer_state, finalized_transitions, insert_mask)

        can_learn = (buffer_state.count >= min_replay_capacity) & (env_step % learn_period_env_steps == 0)

        def _do_learn(args):
            train_state, target_params, buffer_state, learn_rng = args
            sampled_batch, importance_weights, batch_indices = per_sample(
                buffer_state,
                sample_rng,
                batch_size=config.BATCH_SIZE,
                beta=1.0,
                prototype=replay_prototype,
            )
            online_obs_noise_rng, online_next_noise_rng, target_next_noise_rng = jax.random.split(learn_rng, 3)

            def _loss_with_priorities(params: object):
                return _loss(
                    params,
                    target_params,
                    sampled_batch.obs,
                    sampled_batch.action,
                    sampled_batch.reward,
                    sampled_batch.next_obs,
                    sampled_batch.done,
                    online_next_noise_rng,
                    target_next_noise_rng,
                    online_obs_noise_rng,
                    importance_weights,
                )

            (loss, per_example_loss), grads = jax.value_and_grad(_loss_with_priorities, has_aux=True)(
                train_state.params,
            )
            buffer_state = per_update_priorities(buffer_state, batch_indices, per_example_loss)
            return train_state.apply_gradients(grads=grads), buffer_state, loss

        def _skip_learn(args):
            train_state, _target_params, buffer_state, _learn_rng = args
            return train_state, buffer_state, jnp.asarray(0.0)

        train_state, buffer_state, loss = jax.lax.cond(
            can_learn,
            _do_learn,
            _skip_learn,
            (train_state, target_params, buffer_state, learn_rng),
        )

        target_params = jax.lax.cond(
            env_step % target_update_period_env_steps == 0,
            lambda: train_state.params,
            lambda: target_params,
        )

        step_metrics = dict(info)
        step_metrics["loss"] = loss
        step_metrics["max_q"] = jnp.max(rainbow_expected_q_values(action_logits, support))

        next_runner_state = (train_state, target_params, buffer_state, env_state, obs, rng, n_step_accumulator)
        return next_runner_state, step_metrics

    return train_step


def make_train(
    config: RainbowConfig,
    runtime_config: RainbowRuntimeConfig,
    env: object,
    env_params: object | None = None,
):
    env = cast(_EnvLike, env)
    total_env_steps = rainbow_zoo_atari_total_train_env_steps(runtime_config)

    def train(rng):
        network, replay_prototype, runner_state = initialize_train_state(config, env, rng, env_params)
        train_step = make_train_step(config, runtime_config, env, network, replay_prototype, env_params)
        runner_state, metrics = jax.lax.scan(train_step, runner_state, jnp.arange(total_env_steps))
        return {"runner_state": runner_state, "metrics": metrics}

    return train