"""Small behavior tests for named Atari protocol presets."""

from __future__ import annotations

import jax.numpy as jnp
from rl_components.timestep import Timestep
from rl_envs.atari_protocols import nature_dqn_2015


def test_nature_dqn_2015_uses_the_declared_environment_values() -> None:
    """The preset encodes Nature DQN's action and frame preprocessing choices."""
    config = nature_dqn_2015("Pong").env_config

    assert config.game == "Pong"
    assert config.sticky_action_prob == 0.0
    assert config.full_action_space is False
    assert config.noop_max == 30
    assert config.frame_skip == 4
    assert config.terminal_on_life_loss is False
    assert config.max_episode_frames == 108_000


def test_nature_dqn_2015_clips_agent_rewards() -> None:
    """The preset's timestep transform clips rewards to the Nature range."""
    transform = nature_dqn_2015("Pong").timestep_transforms[0]
    timestep = Timestep(
        reward=jnp.asarray(2.0, jnp.float32),
        discount=jnp.asarray(0.9, jnp.float32),
        bootstrap_observation=jnp.asarray([1.0]),
        episode_end=jnp.asarray(False),
        observation=jnp.asarray([2.0]),
    )

    assert float(transform(timestep).reward) == 1.0
