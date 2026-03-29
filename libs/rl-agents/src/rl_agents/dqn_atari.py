from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import optax
from chex import dataclass
from flax.training.train_state import TrainState
from jax_nn.heads import epsilon_greedy_action
from rl_components.atari import JAXAtariConfig, make_atari_adapter
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvProtocol
from rl_components.gymnax_bridge import make_gymnax_compat_env

from rl_agents.dqn import NatureQNetwork, _EnvLike, _infer_nature_observation_layout


@dataclass(frozen=True)
class DQNAtariConfig:
    GAME: str = "pong"
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
    NUM_ITERATIONS: int = 200
    NUM_TRAIN_FRAMES_PER_ITERATION: int = 1_000_000
    EVAL_EXPLORATION_EPSILON: float = 0.05
    ADDITIONAL_DISCOUNT: float = 0.99
    SEED: int = 42

    if TYPE_CHECKING:
        def __init__(
            self,
            *,
            GAME: str = "pong",
            REPLAY_CAPACITY: int = 1_000_000,
            MIN_REPLAY_CAPACITY_FRACTION: float = 0.05,
            BATCH_SIZE: int = 32,
            NUM_ACTION_REPEATS: int = 4,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES: int = 40_000,
            LEARN_PERIOD_FRAMES: int = 16,
            LEARNING_RATE: float = 0.00025,
            OPTIMIZER_EPSILON: float = 0.01 / 32**2,
            RMSPROP_DECAY: float = 0.95,
            RMSPROP_CENTERED: bool = True,
            EXPLORATION_EPSILON_BEGIN: float = 1.0,
            EXPLORATION_EPSILON_END: float = 0.1,
            EXPLORATION_EPSILON_DECAY_FRAME_FRACTION: float = 0.02,
            NUM_ITERATIONS: int = 200,
            NUM_TRAIN_FRAMES_PER_ITERATION: int = 1_000_000,
            EVAL_EXPLORATION_EPSILON: float = 0.05,
            ADDITIONAL_DISCOUNT: float = 0.99,
            SEED: int = 42,
        ) -> None: ...


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


def dqn_zoo_atari_min_replay_capacity(config: DQNAtariConfig) -> int:
    return int(config.REPLAY_CAPACITY * config.MIN_REPLAY_CAPACITY_FRACTION)


def dqn_zoo_atari_learn_period_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.LEARN_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_target_update_period_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_total_train_frames(config: DQNAtariConfig) -> int:
    return config.NUM_ITERATIONS * config.NUM_TRAIN_FRAMES_PER_ITERATION


def dqn_zoo_atari_total_train_env_steps(config: DQNAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(dqn_zoo_atari_total_train_frames(config), config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_exploration_decay_env_steps(config: DQNAtariConfig) -> int:
    decay_frames = int(dqn_zoo_atari_total_train_frames(config) * config.EXPLORATION_EPSILON_DECAY_FRAME_FRACTION)
    return dqn_zoo_atari_frames_to_env_steps(decay_frames, config.NUM_ACTION_REPEATS)


def dqn_zoo_atari_exploration_epsilon(env_step: int, config: DQNAtariConfig) -> float:
    if env_step < 0:
        raise ValueError("env_step must be non-negative.")

    min_replay_capacity = dqn_zoo_atari_min_replay_capacity(config)
    if env_step <= min_replay_capacity:
        return config.EXPLORATION_EPSILON_BEGIN

    decay_env_steps = dqn_zoo_atari_exploration_decay_env_steps(config)
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


def _make_default_env(config: DQNAtariConfig) -> _EnvLike:
    return cast(
        _EnvLike,
        make_gymnax_compat_env(
        cast(
            EnvProtocol[jax.Array, object, jax.Array, None],
            make_atari_adapter(
                JAXAtariConfig(
                    game=config.GAME,
                    frame_skip=config.NUM_ACTION_REPEATS,
                )
            ),
            )
        ),
    )


def _resolve_atari_env(config: DQNAtariConfig, env: object | None, env_params: object | None) -> tuple[_EnvLike, object | None]:
    if env is not None:
        return cast(_EnvLike, env), env_params
    return _make_default_env(config), None


def _sample_uniform_replay_batch(buffer_state: ReplayBufferState, key: jax.Array, batch_size: int):
    indices = jax.random.randint(key, (batch_size,), 0, buffer_state.count)
    return (
        buffer_state.obs[indices],
        buffer_state.actions[indices],
        buffer_state.rewards[indices],
        buffer_state.next_obs[indices],
        buffer_state.dones[indices],
    )


def make_train(config: DQNAtariConfig, env: object | None = None, env_params: object | None = None):
    env, env_params = _resolve_atari_env(config, env, env_params)

    min_replay_capacity = dqn_zoo_atari_min_replay_capacity(config)
    learn_period_env_steps = dqn_zoo_atari_learn_period_env_steps(config)
    target_update_period_env_steps = dqn_zoo_atari_target_update_period_env_steps(config)
    exploration_decay_env_steps = dqn_zoo_atari_exploration_decay_env_steps(config)
    total_env_steps = dqn_zoo_atari_total_train_env_steps(config)

    def train(rng):
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

        def _update_step(runner_state, step_index):
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
                obs_batch, actions, rewards, next_obs, dones = _sample_uniform_replay_batch(
                    buffer_state,
                    sample_rng,
                    config.BATCH_SIZE,
                )
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

            runner_state = (train_state, target_params, buffer_state, env_state, obs, rng)
            return runner_state, step_metrics

        runner_state = (train_state, target_params, buffer_state, env_state, last_obs, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(total_env_steps))
        return {"runner_state": runner_state, "metrics": metrics}

    return train