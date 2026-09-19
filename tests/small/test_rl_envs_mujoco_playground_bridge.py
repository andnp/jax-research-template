"""Small tests for the MuJoCo Playground bridge's truncation handling."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from rl_components.env_protocol import EnvSpec
from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter


@dataclass(frozen=True)
class FakePlaygroundState:
    obs: jax.Array | dict[str, jax.Array]
    reward: jnp.ndarray
    done: jnp.ndarray
    info: dict[str, object]


@dataclass(frozen=True)
class FakeMjModel:
    actuator_ctrlrange: jax.Array | None


class FakePlaygroundEnv:
    """Minimal double matching the playground env/EpisodeWrapper contract."""

    action_size = 1
    observation_size = 1

    def __init__(self, next_state: FakePlaygroundState, mj_model: FakeMjModel | None = None) -> None:
        self._next_state = next_state
        self._mj_model = (
            FakeMjModel(jnp.array([[-jnp.inf, jnp.inf]], dtype=jnp.float32)) if mj_model is None else mj_model
        )

    @property
    def mj_model(self) -> FakeMjModel:
        return self._mj_model

    def reset(self, rng: jax.Array) -> FakePlaygroundState:
        del rng
        return self._next_state

    def step(self, state: FakePlaygroundState, action: jax.Array) -> FakePlaygroundState:
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
        info: dict[str, object],
        expected_terminated: bool,
        expected_truncated: bool,
    ) -> None:
        """Maps Playground done and truncation signals to canonical boundary flags."""
        next_state = FakePlaygroundState(
            obs=jnp.zeros((1,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(1.0, dtype=jnp.float32),
            info=info,
        )
        adapter = MujocoPlaygroundAdapter(FakePlaygroundEnv(next_state))

        result = adapter.step(jax.random.key(0), next_state, jnp.zeros((1,), dtype=jnp.float32))

        assert bool(result.terminated) is expected_terminated
        assert bool(result.truncated) is expected_truncated
        assert result.terminated.dtype == jnp.bool_


class TestMujocoPlaygroundAdapterObservationBoundary:
    def test_dict_observation_is_rejected(self) -> None:
        """Unsupported mapping observations fail at the canonical array boundary."""
        state = FakePlaygroundState(
            obs={"position": jnp.zeros((1,), dtype=jnp.float32)},
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
            info={},
        )
        adapter = MujocoPlaygroundAdapter(FakePlaygroundEnv(state))

        with pytest.raises(TypeError, match="flat array observations"):
            adapter.reset(jax.random.key(0))

    def test_step_info_keeps_only_array_values(self) -> None:
        """Backend metadata is narrowed to the array-only canonical info contract."""
        array_value = jnp.array(1.0, dtype=jnp.float32)
        state = FakePlaygroundState(
            obs=jnp.zeros((1,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
            info={"array": array_value, "scalar": 1.0},
        )
        adapter = MujocoPlaygroundAdapter(FakePlaygroundEnv(state))

        result = adapter.step(jax.random.key(0), state, jnp.zeros((1,), dtype=jnp.float32))

        assert set(result.info) == {"array"}
        assert jnp.array_equal(result.info["array"], array_value)


class TestMujocoPlaygroundAdapterActionBounds:
    """``spec`` runs inside the caller's ``jit``, so its bounds must be constants.

    :func:`rl_components.loop.run` calls ``env.spec`` and ``agent.init`` in the traced
    region, so a ``jnp`` call on the control range is staged unless it is forced to
    evaluate. Staged bounds both crash the unbounded test and reach the agent as
    tracers it cannot compare against.
    """

    @staticmethod
    def _adapter(ctrl_range: jax.Array) -> MujocoPlaygroundAdapter:
        next_state = FakePlaygroundState(
            obs=jnp.zeros((1,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
            info={},
        )
        env = FakePlaygroundEnv(next_state, FakeMjModel(actuator_ctrlrange=ctrl_range))
        return MujocoPlaygroundAdapter(env)

    @staticmethod
    def _spec_read_under_jit(adapter: MujocoPlaygroundAdapter) -> list[EnvSpec]:
        """Collect the spec built while a trace is active, not the returned one."""
        traced: list[EnvSpec] = []

        def record_spec(key: jax.Array):
            return (traced.append(adapter.spec()), key)[1]

        jax.jit(record_spec)(jax.random.key(0))
        return traced

    def test_declared_bounds_are_concrete_inside_a_trace(self) -> None:
        adapter = self._adapter(jnp.array([[-2.0, 3.0]], dtype=jnp.float32))

        (spec,) = self._spec_read_under_jit(adapter)

        assert spec.action_low is not None
        assert spec.action_high is not None
        assert bool(jnp.all(spec.action_low == -2.0))
        assert bool(jnp.all(spec.action_high == 3.0))

    def test_an_unbounded_control_range_is_dropped_inside_a_trace(self) -> None:
        adapter = self._adapter(jnp.array([[-jnp.inf, jnp.inf]], dtype=jnp.float32))

        (spec,) = self._spec_read_under_jit(adapter)

        assert spec.action_low is None
        assert spec.action_high is None
