"""Small tests for the MuJoCo Playground bridge's truncation handling."""

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from rl_components.mujoco_playground_bridge import MujocoPlaygroundAdapter


@dataclass(frozen=True)
class FakePlaygroundState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: dict[str, jnp.ndarray]


class FakePlaygroundEnv:
    """Minimal double matching the playground env/EpisodeWrapper contract."""

    action_size = 1
    observation_size = 1
    mj_model: Any = None

    def __init__(self, next_state: FakePlaygroundState) -> None:
        self._next_state = next_state

    def reset(self, rng: jnp.ndarray) -> FakePlaygroundState:
        del rng
        return self._next_state

    def step(self, state: FakePlaygroundState, action: jnp.ndarray) -> FakePlaygroundState:
        del state, action
        return self._next_state


class TestMujocoPlaygroundAdapterTruncation:
    @pytest.mark.parametrize(
        ("info", "expected_terminated", "expected_truncated"),
        [
            pytest.param({}, True, False, id="no_truncation_key_is_termination"),
            pytest.param({"truncation": jnp.array(1.0, dtype=jnp.float32)}, False, True, id="truncation_flag_set"),
        ],
    )
    def test_step_derives_terminated_and_truncated_from_done_and_truncation(
        self,
        info: dict[str, jnp.ndarray],
        expected_terminated: bool,
        expected_truncated: bool,
    ) -> None:
        next_state = FakePlaygroundState(
            obs=jnp.zeros((1,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(1.0, dtype=jnp.float32),
            info=info,
        )
        adapter = MujocoPlaygroundAdapter(FakePlaygroundEnv(next_state))

        result = adapter.step(jax.random.key(0), cast(Any, next_state), jnp.zeros((1,), dtype=jnp.float32))

        assert bool(result.terminated) is expected_terminated
        assert bool(result.truncated) is expected_truncated
        assert result.terminated.dtype == jnp.bool_
