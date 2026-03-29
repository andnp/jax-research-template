from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax_nn.heads import epsilon_greedy_action
from rl_components.buffers import ReplayBuffer
from rl_components.structs import chex_struct

from rl_agents.dqn import NatureQNetwork, _EnvLike, _infer_nature_observation_layout


@chex_struct(frozen=True, kw_only=True)
class DQNAtariConfig:
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
    EXPLORATION_EPSILON_BEGIN: float = 1.0
    EXPLORATION_EPSILON_END: float = 0.1
    EXPLORATION_EPSILON_DECAY_FRAME_FRACTION: float = 0.02
    ADDITIONAL_DISCOUNT: float = 0.99


@chex_struct(frozen=True, kw_only=True)
class DQNAtariRuntimeConfig:
    TOTAL_TRAIN_ENV_STEPS: int = 50_000_000
    SEED: int = 42
    EVAL_EXPLORATION_EPSILON: float = 0.05


def build_dqn_zoo_atari_rmsprop(config: DQNAtariConfig) -> optax.GradientTransformation:
    return optax.rmsprop(
        learning_rate=config.LEARNING_RATE,
        decay=config.RMSPROP_DECAY,
        eps=config.OPTIMIZER_EPSILON,
        centered=config.RMSPROP_CENTERED,
    )


def dqn_zoo_atari_frames_to_env_steps(frames: int, num_action_repeats: int) -> int:
    if frames < 0:
        raise ValueError("frames must be non-negative.")
    if num_action_repeats <= 0:
        raise ValueError("num_action_repeats must be positive.")
    if frames % num_action_repeats != 0:
        raise ValueError(
            "DQN Zoo frame-counted periods must divide evenly by num_action_repeats when converting to Atari env-step semantics."
        )
    return frames // num_action_repeats


def dqn_atari_runtime_from_dqn_zoo(
    config: DQNAtariConfig,
    *,
    num_iterations: int = 200,
    num_train_frames_per_iteration: int = 1_000_000,
    seed: int = 42,
    eval_exploration_epsilon: float = 0.05,
):
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative.")
    if num_train_frames_per_iteration < 0:
        raise ValueError("num_train_frames_per_iteration must be non-negative.")

    # DQN Zoo counts emulator frames, while this Atari runtime trains in env steps.
    # With action repeat, one env.step advances NUM_ACTION_REPEATS frames, so frame-based schedules must divide by that repeat count.
    total_train_frames = num_iterations * num_train_frames_per_iteration
    return DQNAtariRuntimeConfig(
        TOTAL_TRAIN_ENV_STEPS=dqn_zoo_atari_frames_to_env_steps(total_train_frames, config.NUM_ACTION_REPEATS),
        SEED=seed,
        EVAL_EXPLORATION_EPSILON=eval_exploration_epsilon,
    )


def dqn_zoo_atari_min_replay_capacity(config: DQNAtariConfig) -> int:
    return int(config.REPLAY_CAPACITY * config.MIN_REPLAY_CAPACITY_FRACTION)


def dqn_zoo_atari_learn_period_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.LEARN_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_target_update_period_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_total_train_frames(config: DQNAtariConfig, runtime_config: DQNAtariRuntimeConfig) -> int:
    total_train_env_steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
    return total_train_env_steps * config.NUM_ACTION_REPEATS


def dqn_zoo_atari_total_train_env_steps(runtime_config: DQNAtariRuntimeConfig) -> int:
    if runtime_config.TOTAL_TRAIN_ENV_STEPS < 0:
        raise ValueError("TOTAL_TRAIN_ENV_STEPS must be non-negative.")
    return runtime_config.TOTAL_TRAIN_ENV_STEPS


def dqn_zoo_atari_exploration_decay_env_steps(config: DQNAtariConfig, runtime_config: DQNAtariRuntimeConfig) -> int:
    decay_frames = int(
        dqn_zoo_atari_total_train_frames(config, runtime_config) * config.EXPLORATION_EPSILON_DECAY_FRAME_FRACTION
    )
    return dqn_zoo_atari_frames_to_env_steps(decay_frames, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_exploration_epsilon(
    env_step: int,
    config: DQNAtariConfig,
    runtime_config: DQNAtariRuntimeConfig,
) -> float:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")

    min_replay_capacity = dqn_zoo_atari_min_replay_capacity(config)
    if env_step <= min_replay_capacity:
        return config.EXPLORATION_EPSILON_BEGIN

    decay_env_steps = dqn_zoo_atari_exploration_decay_env_steps(config, runtime_config)
    if decay_env_steps <= 0:
        raise ValueError("exploration decay must span at least one env step.")

    elapsed_decay_steps = min(env_step - min_replay_capacity, decay_env_steps)
    if elapsed_decay_steps >= decay_env_steps:
        return config.EXPLORATION_EPSILON_END

    progress = elapsed_decay_steps / decay_env_steps
    return config.EXPLORATION_EPSILON_BEGIN + (
        config.EXPLORATION_EPSILON_END - config.EXPLORATION_EPSILON_BEGIN
    ) * progress


def dqn_zoo_atari_should_learn(env_step: int, replay_size: int, config: DQNAtariConfig) -> bool:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")
    if replay_size < 0:
        raise ValueError("replay_size must be non-negative.")
    if replay_size < dqn_zoo_atari_min_replay_capacity(config):
        return False
    return env_step % dqn_zoo_atari_learn_period_env_steps(config) == 0


def initialize_train_state(
    config: DQNAtariConfig,
    env: object,
    rng: jax.Array,
    env_params: object | None = None,
):
    env = cast(_EnvLike, env)

    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    observation_shape = tuple(observation_space.shape)
    network = NatureQNetwork(
        action_dim=action_space.n,
        observation_layout=_infer_nature_observation_layout(observation_shape),
    )

    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros(observation_shape, dtype=observation_space.dtype)
    params = network.init(init_rng, init_x)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=build_dqn_zoo_atari_rmsprop(config),
    )
    target_params = params

    buffer = ReplayBuffer(
        config.REPLAY_CAPACITY,
        observation_shape,
        (),
        jnp.int32,
        observation_space.dtype,
    )
    buffer_state = buffer.init()

    rng, reset_rng = jax.random.split(rng)
    last_obs, env_state = env.reset(reset_rng, env_params)
    runner_state = (train_state, target_params, buffer_state, env_state, last_obs, rng)
    return network, buffer, runner_state


def make_train_step(
    config: DQNAtariConfig,
    runtime_config: DQNAtariRuntimeConfig,
    env: object,
    network: NatureQNetwork,
    buffer: ReplayBuffer,
    env_params: object | None = None,
):
    env = cast(_EnvLike, env)

    min_replay_capacity = dqn_zoo_atari_min_replay_capacity(config)
    learn_period_env_steps = dqn_zoo_atari_learn_period_env_steps(config)
    target_update_period_env_steps = dqn_zoo_atari_target_update_period_env_steps(config)
    exploration_decay_env_steps = dqn_zoo_atari_exploration_decay_env_steps(config, runtime_config)

    def _exploration_epsilon(env_step: jax.Array) -> jax.Array:
        warmup_complete = env_step > min_replay_capacity
        elapsed_decay_steps = jnp.minimum(jnp.maximum(env_step - min_replay_capacity, 0), exploration_decay_env_steps)
        progress = elapsed_decay_steps / exploration_decay_env_steps
        decayed = config.EXPLORATION_EPSILON_BEGIN + (
            config.EXPLORATION_EPSILON_END - config.EXPLORATION_EPSILON_BEGIN
        ) * progress
        return jnp.where(warmup_complete, decayed, config.EXPLORATION_EPSILON_BEGIN)

    def _loss(params, target_params, obs, actions, rewards, next_obs, dones):
        q_values = network.apply(params, obs)
        q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)

        next_q_values = network.apply(target_params, next_obs)
        next_q_max = jnp.max(next_q_values, axis=-1)
        targets = rewards + config.ADDITIONAL_DISCOUNT * next_q_max * (1.0 - dones)
        td_error = q_action - jax.lax.stop_gradient(targets)
        return jnp.mean(jnp.square(td_error))

    def train_step(runner_state, step_index):
        train_state, target_params, buffer_state, env_state, last_obs, rng = runner_state

        env_step = step_index + 1
        epsilon = _exploration_epsilon(env_step)

        rng, action_rng, step_rng, sample_rng = jax.random.split(rng, 4)
        q_values = network.apply(train_state.params, last_obs)
        action = epsilon_greedy_action(q_values, epsilon, key=action_rng)

        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
        buffer_state = buffer.add(
            buffer_state,
            last_obs[None, ...],
            action[None, ...],
            reward[None, ...],
            obs[None, ...],
            done[None, ...],
        )

        can_learn = (buffer_state.count >= min_replay_capacity) & (env_step % learn_period_env_steps == 0)

        def _do_learn(args):
            train_state, target_params, buffer_state = args
            indices = jax.random.randint(
                sample_rng,
                (config.BATCH_SIZE,),
                0,
                buffer_state.count,
            )
            obs_batch = buffer_state.obs[indices]
            actions = buffer_state.actions[indices]
            rewards = buffer_state.rewards[indices]
            next_obs = buffer_state.next_obs[indices]
            dones = buffer_state.dones[indices]
            loss, grads = jax.value_and_grad(_loss)(
                train_state.params,
                target_params,
                obs_batch,
                actions,
                rewards,
                next_obs,
                dones,
            )
            return train_state.apply_gradients(grads=grads), loss

        def _skip_learn(args):
            train_state, _target_params, _buffer_state = args
            return train_state, jnp.asarray(0.0)

        train_state, loss = jax.lax.cond(
            can_learn,
            _do_learn,
            _skip_learn,
            (train_state, target_params, buffer_state),
        )

        target_params = jax.lax.cond(
            env_step % target_update_period_env_steps == 0,
            lambda: train_state.params,
            lambda: target_params,
        )

        step_metrics = dict(info)
        step_metrics["epsilon"] = epsilon
        step_metrics["loss"] = loss

        next_runner_state = (train_state, target_params, buffer_state, env_state, obs, rng)
        return next_runner_state, step_metrics

    return train_step


def make_train(
    config: DQNAtariConfig,
    runtime_config: DQNAtariRuntimeConfig,
    env: object,
    env_params: object | None = None,
):
    env = cast(_EnvLike, env)
    total_env_steps = dqn_zoo_atari_total_train_env_steps(runtime_config)

    def train(rng):
        network, buffer, runner_state = initialize_train_state(config, env, rng, env_params)
        train_step = make_train_step(config, runtime_config, env, network, buffer, env_params)
        runner_state, metrics = jax.lax.scan(train_step, runner_state, jnp.arange(total_env_steps))
        return {"runner_state": runner_state, "metrics": metrics}

    return train