import os
from pathlib import Path
from typing import cast

import jax
import pytest
from rl_agents.dqn_atari import DQNAtariConfig, make_train
from rl_components.atari import JAXAtariConfig, make_atari_adapter
from rl_components.env_protocol import EnvProtocol
from rl_components.gymnax_bridge import make_gymnax_compat_env


def _jaxatari_assets_dir() -> Path:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "jaxatari"
    return Path.home() / ".local" / "share" / "jaxatari"


def _require_real_smoke() -> None:
    if os.environ.get("JAXATARI_RUN_SMOKE") != "1":
        pytest.skip("set JAXATARI_RUN_SMOKE=1 to run the real JAXAtari DQN smoke test")
    assets_dir = _jaxatari_assets_dir()
    if not (assets_dir / "sprites" / "pong").exists():
        pytest.skip(f"install JAXAtari assets under {assets_dir} before running the real smoke test")


def test_real_dqn_nature_path_on_jaxatari_smoke() -> None:
    _require_real_smoke()

    config = DQNAtariConfig(
        REPLAY_CAPACITY=16,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=4,
        LEARN_PERIOD_FRAMES=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
        NUM_ITERATIONS=1,
        NUM_TRAIN_FRAMES_PER_ITERATION=16,
    )
    env = make_gymnax_compat_env(
        cast(EnvProtocol[jax.Array, object, jax.Array, None], make_atari_adapter(JAXAtariConfig(game="pong")))
    )

    train = make_train(config, env=env, env_params=None)
    out = jax.jit(train)(jax.random.key(0))

    metrics = out["metrics"]
    assert metrics["returned_episode"].shape == (4,)
    assert metrics["returned_episode_returns"].shape == (4,)