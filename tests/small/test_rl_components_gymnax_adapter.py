"""Small tests for the gymnax → EnvProtocol adapter with correct trunc/term signals."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from rl_components.gymnax_adapter import GymnaxEnvAdapter, make_gymnax_env

# ---------------------------------------------------------------------------
# Minimal fake gymnax environment
# ---------------------------------------------------------------------------

@struct.dataclass
class FakeEnvState:
    time: int


@struct.dataclass
class FakeEnvParams:
    max_steps_in_episode: int = 10
    terminate_at: int = 999  # step at which env sends done=True; 999 = never in tests


class FakeGymnaxEnv:
    """Gymnax-style env with controllable termination for deterministic tests."""

    name = "fake"

    @property
    def default_params(self) -> FakeEnvParams:
        return FakeEnvParams()

    def observation_space(self, params: FakeEnvParams) -> Any:
        from gymnax.environments import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32)

    def action_space(self, params: FakeEnvParams) -> Any:
        from gymnax.environments import spaces
        return spaces.Discrete(2)

    def reset_env(self, key: jax.Array, params: FakeEnvParams) -> tuple[jax.Array, FakeEnvState]:
        del key
        return jnp.zeros((2,), dtype=jnp.float32), FakeEnvState(time=0)  # type: ignore

    def step_env(
        self,
        key: jax.Array,
        state: FakeEnvState,
        action: Any,
        params: FakeEnvParams,
    ) -> tuple[jax.Array, FakeEnvState, jax.Array, jax.Array, dict]:
        del key, action
        new_state = FakeEnvState(time=state.time + 1)  # type: ignore
        obs = jnp.full((2,), new_state.time, dtype=jnp.float32)
        reward = jnp.array(1.0)
        done = jnp.array(new_state.time >= params.terminate_at)
        return obs, new_state, reward, done, {}


def _make_adapter(max_steps: int = 5, terminate_at: int = 999) -> GymnaxEnvAdapter:
    env = FakeGymnaxEnv()
    params = FakeEnvParams(max_steps_in_episode=max_steps + 1, terminate_at=terminate_at)  # type: ignore
    return GymnaxEnvAdapter(env, max_steps=max_steps, default_params=params)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGymnaxEnvAdapterSignals:
    def test_normal_step_has_neither_signal(self) -> None:
        adapter = _make_adapter(max_steps=5)
        key = jax.random.key(0)
        reset = adapter.reset(key)
        step = adapter.step(key, reset.state, jnp.array(0))

        assert not bool(step.terminated)
        assert not bool(step.truncated)

    def test_truncation_fires_at_max_steps(self) -> None:
        adapter = _make_adapter(max_steps=3)
        key = jax.random.key(0)
        reset = adapter.reset(key)
        state = reset.state

        for _ in range(2):
            result = adapter.step(key, state, jnp.array(0))
            assert not bool(result.terminated)
            assert not bool(result.truncated)
            state = result.state

        final = adapter.step(key, state, jnp.array(0))
        assert not bool(final.terminated)
        assert bool(final.truncated)

    def test_termination_fires_before_max_steps(self) -> None:
        adapter = _make_adapter(max_steps=10, terminate_at=2)
        key = jax.random.key(0)
        reset = adapter.reset(key)
        state = reset.state

        first = adapter.step(key, state, jnp.array(0))
        assert not bool(first.terminated)
        assert not bool(first.truncated)

        second = adapter.step(key, first.state, jnp.array(0))
        assert bool(second.terminated)
        assert not bool(second.truncated)

    def test_both_signals_false_mid_episode(self) -> None:
        adapter = _make_adapter(max_steps=10, terminate_at=999)
        key = jax.random.key(0)
        reset = adapter.reset(key)
        state = reset.state

        for _ in range(5):
            result = adapter.step(key, state, jnp.array(0))
            assert not bool(result.terminated)
            assert not bool(result.truncated)
            state = result.state


class TestGymnaxEnvAdapterAutoReset:
    def test_state_resets_after_truncation(self) -> None:
        adapter = _make_adapter(max_steps=3)
        key = jax.random.key(0)
        reset = adapter.reset(key)
        state = reset.state

        for _ in range(2):
            state = adapter.step(key, state, jnp.array(0)).state

        truncated_step = adapter.step(key, state, jnp.array(0))
        assert bool(truncated_step.truncated)
        # After auto-reset, time should be 0
        assert int(truncated_step.state.time) == 0

    def test_state_resets_after_termination(self) -> None:
        adapter = _make_adapter(max_steps=10, terminate_at=2)
        key = jax.random.key(0)
        reset = adapter.reset(key)
        state = adapter.step(key, reset.state, jnp.array(0)).state

        terminal_step = adapter.step(key, state, jnp.array(0))
        assert bool(terminal_step.terminated)
        assert int(terminal_step.state.time) == 0

    def test_observation_after_truncation_is_reset_obs(self) -> None:
        adapter = _make_adapter(max_steps=3)
        key = jax.random.key(0)
        reset_obs = adapter.reset(key).observation
        state = adapter.reset(key).state

        for _ in range(2):
            state = adapter.step(key, state, jnp.array(0)).state

        truncated_step = adapter.step(key, state, jnp.array(0))
        # After truncation the returned obs is the reset obs (all zeros for FakeGymnaxEnv)
        assert bool(jnp.all(truncated_step.observation == reset_obs))


class TestGymnaxEnvAdapterSpec:
    def test_spec_discrete_action_space(self) -> None:
        adapter = _make_adapter()
        spec = adapter.spec()

        assert spec.id == "fake"
        assert spec.observation_shape == (2,)
        assert spec.action_shape == ()
        assert spec.num_actions == 2
        assert spec.action_dtype == jnp.int32

    def test_gymnax_max_steps_is_bumped_to_t_plus_one(self) -> None:
        adapter = _make_adapter(max_steps=7)
        gp = adapter._gymnax_params(None)
        assert gp.max_steps_in_episode == 8


class TestMakeGymnaxEnv:
    def test_factory_creates_adapter(self) -> None:
        adapter = make_gymnax_env("CartPole-v1")
        assert isinstance(adapter, GymnaxEnvAdapter)

    def test_factory_uses_env_default_max_steps_when_not_specified(self) -> None:
        adapter = make_gymnax_env("CartPole-v1")
        # CartPole-v1 default is 500
        assert adapter._max_steps == 500

    def test_factory_respects_explicit_max_steps(self) -> None:
        adapter = make_gymnax_env("CartPole-v1", max_steps=200)
        assert adapter._max_steps == 200
        gp = adapter._gymnax_params(None)
        assert gp.max_steps_in_episode == 201

    def test_factory_adapter_runs_a_step(self) -> None:
        adapter = make_gymnax_env("CartPole-v1", max_steps=5)
        key = jax.random.key(42)
        reset = adapter.reset(key)
        step = adapter.step(key, reset.state, jnp.array(0))

        assert step.observation.shape == (4,)
        assert step.reward.shape == ()
        assert not bool(step.truncated)
        assert not bool(step.terminated)
