"""Named Atari environment and agent-input preprocessing protocols."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from rl_components.timestep_pipeline import TimestepTransform, clip_reward

from rl_envs.atari_ale import AleAtariConfig


@dataclass(frozen=True)
class AtariProtocol:
    env_config: AleAtariConfig
    timestep_transforms: tuple[TimestepTransform[jax.Array], ...]


def nature_dqn_2015(game: str) -> AtariProtocol:
    """Return the Nature DQN (2015) Atari protocol.

    The 108,000-frame episode cap follows ALE v5 and DQN Zoo because Nature DQN does not
    specify a training cap. Nature DQN treated life loss as terminal for learning while
    the game continued; ``Timestep`` has no lives signal, so this preset terminates only
    at game over. TD-error clipping belongs to the agent (``DQNAtariConfig.GRAD_ERROR_BOUND``),
    not to this environment and reward-preprocessing preset.
    """
    return AtariProtocol(
        env_config=AleAtariConfig(
            game=game,
            frame_skip=4,
            max_episode_frames=108_000,
            terminal_on_life_loss=False,
            sticky_action_prob=0.0,
            full_action_space=False,
            noop_max=30,
        ),
        timestep_transforms=(clip_reward(-1.0, 1.0),),
    )
