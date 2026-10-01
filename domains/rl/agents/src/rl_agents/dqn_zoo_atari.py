"""Definitions DQN Zoo Atari agents share."""

import jax.numpy as jnp
import optax
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class DQNZooAtariConfig:
    replay_capacity: int = 1_000_000  # DQN: ~14 GB at 1M, two uint8 84x84 frames per slot.
    min_replay_capacity_fraction: float = 0.05
    batch_size: int = 32
    num_action_repeats: int = 4
    target_network_update_period_frames: int = 40_000
    learn_period_frames: int = 16
    learning_rate: float = 0.00025
    optimizer_epsilon: float = 0.01 / 32**2
    rmsprop_decay: float = 0.95
    rmsprop_centered: bool = True


def build_dqn_zoo_atari_rmsprop(config: DQNZooAtariConfig) -> optax.GradientTransformation:
    # Injected so a sweep can overwrite each scalar in opt_state instead of leaking one arm's value into every arm's compiled transform.
    return optax.inject_hyperparams(optax.rmsprop, static_args="centered")(
        learning_rate=jnp.asarray(config.learning_rate, jnp.float32),
        decay=jnp.asarray(config.rmsprop_decay, jnp.float32),
        eps=jnp.asarray(config.optimizer_epsilon, jnp.float32),
        centered=config.rmsprop_centered,
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


def dqn_zoo_atari_min_replay_capacity(config: DQNZooAtariConfig) -> int:
    return int(config.replay_capacity * config.min_replay_capacity_fraction)


def dqn_zoo_atari_learn_period_env_steps(config: DQNZooAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.learn_period_frames, config.num_action_repeats)


def dqn_zoo_atari_target_update_period_env_steps(config: DQNZooAtariConfig) -> int:
    return dqn_zoo_atari_frames_to_env_steps(config.target_network_update_period_frames, config.num_action_repeats)
