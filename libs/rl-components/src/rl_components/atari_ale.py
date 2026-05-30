from __future__ import annotations

import ale_py  # noqa: F401  # registers ALE namespace with gymnasium
import gymnasium
from gymnasium.wrappers import AtariPreprocessing

from rl_components.frame_stack import FrameStackWrapper
from rl_components.python_env_bridge import PythonEnvBridge
from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class AleAtariConfig:
    game: str
    frame_stack: int = 4
    frame_skip: int = 4
    grayscale: bool = True
    screen_size: int = 84
    terminal_on_life_loss: bool = True
    scale_obs: bool = False


def _make_ale_env(config: AleAtariConfig) -> gymnasium.Env:
    env = gymnasium.make(
        f"ALE/{config.game}-v5",
        frameskip=1,  # frame skip is handled by AtariPreprocessing
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=None,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=config.grayscale,
        grayscale_newaxis=True,  # adds channel dim: (H, W, 1)
        scale_obs=config.scale_obs,
    )
    return env


def make_atari_adapter(config: AleAtariConfig) -> FrameStackWrapper:
    """Create an ALE Atari adapter conforming to EnvProtocol.

    Returns a FrameStackWrapper around a PythonEnvBridge wrapping an ALE gymnasium env.
    The StateT is FrameStackState[jnp.uint8[N]] where N is the serialized ALE state size.
    """
    bridge = PythonEnvBridge(
        make_env=lambda: _make_ale_env(config),
        env_id=f"ale:{config.game}",
    )
    return FrameStackWrapper(bridge, n_frames=config.frame_stack)
