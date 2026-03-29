import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_components.atari import JAXAtariConfig, make_atari_adapter


def _jaxatari_assets_dir():
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "jaxatari"
    return Path.home() / ".local" / "share" / "jaxatari"


def _require_real_smoke():
    if os.environ.get("JAXATARI_RUN_SMOKE") != "1":
        pytest.skip("set JAXATARI_RUN_SMOKE=1 to run the real JAXAtari smoke test")
    assets_dir = _jaxatari_assets_dir()
    if not (assets_dir / "sprites" / "pong").exists():
        pytest.skip(f"install JAXAtari assets under {assets_dir} before running the real smoke test")
    return assets_dir


def test_real_jaxatari_adapter_smoke():
    assets_dir = _require_real_smoke()
    pong_assets = sorted((assets_dir / "sprites" / "pong").glob("*.npy"))
    sprites = np.load(pong_assets[0])

    adapter = make_atari_adapter(JAXAtariConfig(game="pong"))
    reset = adapter.reset(jax.random.key(0))
    transition = adapter.step(jax.random.key(1), reset.state, jnp.array(2, dtype=jnp.int32))

    assert pong_assets
    assert sprites.size > 0
    assert reset.observation.shape == (4, 84, 84, 1)
    assert {"returned_episode", "returned_episode_lengths", "returned_episode_returns", "time"} <= set(transition.info)
