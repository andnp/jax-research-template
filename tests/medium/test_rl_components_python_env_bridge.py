"""Medium tests for PythonEnvBridge terminal observation semantics."""

from __future__ import annotations

from typing import Literal, override

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_components.frame_stack import FrameStackWrapper
from rl_components.gymnax_bridge import make_gymnax_compat_env
from rl_components.python_env_bridge import (
    FINAL_OBSERVATION_INFO_KEY,
    FINAL_OBSERVATION_VALID_INFO_KEY,
    PythonEnvBridge,
)


class _FakeALE:
    def __init__(self, env: "_FakeGymALE") -> None:
        self._env = env

    def cloneState(self) -> tuple[int, int]:
        return self._env._episode_start, self._env._episode_steps

    def restoreState(self, state: object, /) -> None:
        if not isinstance(state, tuple) or len(state) != 2 or not all(isinstance(value, int) for value in state):
            raise TypeError("unexpected fake ALE state")
        self._env._episode_start, self._env._episode_steps = state


class _FakeGymALE(gymnasium.Env[np.ndarray, int]):
    observation_space = gymnasium.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
    action_space = gymnasium.spaces.Discrete(2)

    def __init__(self, end_kind: Literal["terminated", "truncated"]) -> None:
        self.ale = _FakeALE(self)
        self._end_kind = end_kind
        self._reset_count = 0
        self._episode_start = 0
        self._episode_steps = 0

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        del seed, options
        self._reset_count += 1
        self._episode_start = self._reset_count * 10
        self._episode_steps = 0
        return np.array([self._episode_start], dtype=np.uint8), {}

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._episode_steps += 1
        observation = np.array([self._episode_start + self._episode_steps], dtype=np.uint8)
        done = self._episode_steps == 2
        return (
            observation,
            float(action),
            done if self._end_kind == "terminated" else False,
            done if self._end_kind == "truncated" else False,
            {},
        )


@pytest.mark.parametrize("end_kind", ["terminated", "truncated"])
def test_bridge_preserves_post_reset_observation_and_exposes_terminal_frame(
    end_kind: Literal["terminated", "truncated"],
) -> None:
    """Verify terminal frames survive the bridge's fused autoreset."""
    bridge = PythonEnvBridge(lambda: _FakeGymALE(end_kind))
    reset = bridge.reset(jax.random.key(0))

    first = bridge.step(jax.random.key(1), reset.state, jnp.array(0, dtype=jnp.int32))
    terminal = bridge.step(jax.random.key(2), first.state, jnp.array(0, dtype=jnp.int32))
    next_step = bridge.step(jax.random.key(3), terminal.state, jnp.array(0, dtype=jnp.int32))

    assert int(first.observation[0]) == 31
    assert int(first.info[FINAL_OBSERVATION_INFO_KEY][0]) == 31
    assert bool(first.info[FINAL_OBSERVATION_VALID_INFO_KEY]) is False
    assert int(terminal.observation[0]) == 40
    assert int(terminal.info[FINAL_OBSERVATION_INFO_KEY][0]) == 32
    assert bool(terminal.info[FINAL_OBSERVATION_VALID_INFO_KEY]) is True
    assert int(next_step.observation[0]) == 41
    assert bool(next_step.info[FINAL_OBSERVATION_VALID_INFO_KEY]) is False


def test_frame_stack_rollout_uses_reset_observation_and_preserves_terminal_info() -> None:
    """Verify stacked rollouts act from reset frames after a terminal step."""
    bridge = PythonEnvBridge(lambda: _FakeGymALE("terminated"))
    env = make_gymnax_compat_env(FrameStackWrapper(bridge, n_frames=2))

    def rollout(key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        observation, state = env.reset(key, None)

        def step(
            carry: tuple[object, jax.Array], _: jax.Array
        ) -> tuple[tuple[object, jax.Array], tuple[jax.Array, dict[str, jax.Array]]]:
            state, last_observation = carry
            action = jnp.asarray(last_observation[-1, 0] % 2, dtype=jnp.int32)
            next_observation, next_state, _reward, _done, info = env.step(key, state, action, None)
            return (next_state, next_observation), (next_observation, info)

        (_, _), outputs = jax.lax.scan(step, (state, observation), jnp.arange(3))
        return outputs

    observations, info = jax.jit(rollout)(jax.random.key(0))

    assert observations[:, -1, 0].tolist() == [31, 40, 41]
    assert info[FINAL_OBSERVATION_VALID_INFO_KEY].tolist() == [False, True, False]
    assert info[FINAL_OBSERVATION_INFO_KEY][1, :, 0].tolist() == [31, 32]
