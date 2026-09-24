"""Machado et al. 2018 Atari preprocessing exposed through the environment protocol.

The default sticky-action probability, action set and no-op behavior follow the evaluation
protocol in Machado et al. (2018).
"""

from __future__ import annotations

from typing import Any, SupportsFloat, override

import ale_py  # noqa: F401  # registers ALE namespace with gymnasium
import gymnasium
import jax
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.wrappers import AtariPreprocessing
from rl_components.structs import chex_struct

from rl_envs.frame_stack import FrameStackWrapper
from rl_envs.python_env_bridge import PythonEnvBridge


class _AtariPreprocessing(AtariPreprocessing):
    @override
    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        first_frame = self.ale.getEpisodeFrameNumber()
        observation, reward, terminated, truncated, info = super().step(action)
        frames_advanced = self.ale.getEpisodeFrameNumber() - first_frame

        if terminated or truncated:
            # AtariPreprocessing exits before capturing buffers on its terminal sub-step.
            if self.grayscale_obs:
                self.ale.getScreenGrayscale(self.obs_buffer[0])
            else:
                self.ale.getScreenRGB(self.obs_buffer[0])
            if frames_advanced != self.frame_skip:
                self.obs_buffer[1].fill(0)
            observation = self._get_obs()

        return observation, reward, terminated, truncated, info


@chex_struct(frozen=True)
class AleAtariConfig:
    """Atari environment configuration, defaulting to Machado et al. (2018)."""

    game: str
    frame_stack: int = 4
    frame_skip: int = 4
    grayscale: bool = True
    screen_size: int = 84
    terminal_on_life_loss: bool = False
    scale_obs: bool = False
    sticky_action_prob: float = 0.25
    full_action_space: bool = True
    noop_max: int = 0


def _make_ale_env(config: AleAtariConfig) -> _AtariPreprocessing:
    env = gymnasium.make(
        f"ALE/{config.game}-v5",
        frameskip=1,  # frame skip is handled by AtariPreprocessing
        repeat_action_probability=config.sticky_action_prob,
        full_action_space=config.full_action_space,
        render_mode=None,
    )
    env = _AtariPreprocessing(
        env,
        noop_max=config.noop_max,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=config.grayscale,
        grayscale_newaxis=True,  # adds channel dim: (H, W, 1)
        scale_obs=config.scale_obs,
    )
    return env


def make_atari_adapter(config: AleAtariConfig) -> FrameStackWrapper[jax.Array, jax.Array, jax.Array, None]:
    """Create an ALE Atari adapter conforming to EnvProtocol.

    Returns a FrameStackWrapper around a PythonEnvBridge wrapping an ALE gymnasium env.
    The StateT is FrameStackState[jnp.uint8[N]] where N is the serialized ALE state size.
    """
    bridge = PythonEnvBridge(
        make_env=lambda: _make_ale_env(config),
        env_id=f"ale:{config.game}",
    )
    return FrameStackWrapper(bridge, n_frames=config.frame_stack)
