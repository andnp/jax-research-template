"""Definitions DQN Zoo Atari agents share."""

import jax.numpy as jnp
import optax
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class DQNZooAtariConfig:
    REPLAY_CAPACITY: int = 1_000_000  # DQN: ~14 GB at 1M, two uint8 84x84 frames per slot.
    MIN_REPLAY_CAPACITY_FRACTION: float = 0.05
    BATCH_SIZE: int = 32
    NUM_ACTION_REPEATS: int = 4
    TARGET_NETWORK_UPDATE_PERIOD_FRAMES: int = 40_000
    LEARN_PERIOD_FRAMES: int = 16
    LEARNING_RATE: float = 0.00025
    OPTIMIZER_EPSILON: float = 0.01 / 32**2
    RMSPROP_DECAY: float = 0.95
    RMSPROP_CENTERED: bool = True


def build_dqn_zoo_atari_rmsprop(config: DQNZooAtariConfig) -> optax.GradientTransformation:
    # Injected so a sweep can overwrite each scalar in opt_state instead of leaking one arm's value into every arm's compiled transform.
    return optax.inject_hyperparams(optax.rmsprop, static_args="centered")(
        learning_rate=jnp.asarray(config.LEARNING_RATE, jnp.float32),
        decay=jnp.asarray(config.RMSPROP_DECAY, jnp.float32),
        eps=jnp.asarray(config.OPTIMIZER_EPSILON, jnp.float32),
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
