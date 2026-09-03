"""Small tests pinning the Brax time-limit truncation mapping."""

from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import pytest
import rl_components.brax as brax_module
from rl_components.brax import BraxAdapter, BraxConfig


@dataclass(frozen=True)
class FakeBraxState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: dict[str, object]


@dataclass(frozen=True)
class FakeBraxActuator:
    ctrl_range: jnp.ndarray


@dataclass(frozen=True)
class FakeBraxSystem:
    actuator: FakeBraxActuator


class FakeBraxEnv:
    action_size = 1
    observation_size = 2
    sys = FakeBraxSystem(actuator=FakeBraxActuator(ctrl_range=jnp.array([[-1.0, 1.0]], dtype=jnp.float32)))

    def __init__(self, done: float, truncation: float) -> None:
        self._done = done
        self._truncation = truncation

    def reset(self, key: chex.PRNGKey) -> FakeBraxState:
        del key
        return FakeBraxState(
            obs=jnp.zeros((2,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
            info={"truncation": jnp.array(0.0, dtype=jnp.float32)},
        )

    def step(self, state: FakeBraxState, action: jnp.ndarray) -> FakeBraxState:
        del state, action
        return FakeBraxState(
            obs=jnp.ones((2,), dtype=jnp.float32),
            reward=jnp.array(1.0, dtype=jnp.float32),
            done=jnp.array(self._done, dtype=jnp.float32),
            info={"truncation": jnp.array(self._truncation, dtype=jnp.float32)},
        )


class TestBraxAdapterTruncationMapping:
    @pytest.mark.parametrize(
        ("done", "truncation", "expected_terminated", "expected_truncated"),
        [
            pytest.param(1.0, 1.0, False, True, id="time_limit"),
            pytest.param(1.0, 0.0, True, False, id="genuine_termination"),
            pytest.param(0.0, 0.0, False, False, id="continuation"),
        ],
    )
    def test_step_maps_done_and_truncation_to_terminated_and_truncated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        done: float,
        truncation: float,
        expected_terminated: bool,
        expected_truncated: bool,
    ) -> None:
        monkeypatch.setattr(brax_module, "_make_brax_env", lambda config: FakeBraxEnv(done, truncation))
        adapter = BraxAdapter(BraxConfig(env_name="fake"))
        state = adapter.reset(jax.random.key(0)).state

        step = adapter.step(jax.random.key(1), state, jnp.zeros((1,), dtype=jnp.float32))

        assert bool(step.terminated) is expected_terminated
        assert bool(step.truncated) is expected_truncated
        assert step.terminated.dtype == jnp.bool_
