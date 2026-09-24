"""Small behavior tests for timestep preprocessing transforms."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from rl_components.timestep import Timestep
from rl_components.timestep_pipeline import clip_reward


@pytest.mark.parametrize(
    ("reward", "expected"),
    [
        pytest.param(-0.5, -0.5, id="inside"),
        pytest.param(-1.0, -1.0, id="lower-bound"),
        pytest.param(1.0, 1.0, id="upper-bound"),
        pytest.param(-2.0, -1.0, id="below-range"),
        pytest.param(2.0, 1.0, id="above-range"),
    ],
)
def test_clip_reward_bounds_only_the_reward(reward: float, expected: float) -> None:
    """In-range and boundary rewards stay fixed while out-of-range rewards clip."""
    timestep = Timestep(
        reward=jnp.asarray(reward, jnp.float32),
        discount=jnp.asarray(0.75, jnp.float32),
        bootstrap_observation=jnp.asarray([11.0, 12.0]),
        episode_end=jnp.asarray(True),
        observation=jnp.asarray([21.0, 22.0]),
    )

    transformed = clip_reward(-1.0, 1.0)(timestep)

    assert float(transformed.reward) == expected
    np.testing.assert_array_equal(transformed.discount, timestep.discount)
    np.testing.assert_array_equal(transformed.bootstrap_observation, timestep.bootstrap_observation)
    np.testing.assert_array_equal(transformed.episode_end, timestep.episode_end)
    np.testing.assert_array_equal(transformed.observation, timestep.observation)
