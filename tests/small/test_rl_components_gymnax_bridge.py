"""Small tests for the canonical-to-Gymnax compatibility bridge."""

from dataclasses import dataclass
from typing import cast

import chex
import jax
import jax.numpy as jnp
import rl_components.brax as brax_module
from rl_components.brax import BraxAdapter, BraxConfig
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gymnax_bridge import GymnaxCompatibilityBridge, GymnaxDiscreteSpace, GymnaxSpace


class DummyDiscreteEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="discrete",
            observation_shape=(4,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=3,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros((4,), dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        return EnvStep(
            observation=jnp.full((4,), next_state, dtype=jnp.float32),
            state=next_state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(True),
            info={"episode_length": next_state},
        )


class DummyContinuousEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="continuous",
            observation_shape=(3,),
            action_shape=(2,),
            observation_dtype=jnp.float32,
            action_dtype=jnp.float32,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros((3,), dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        del key, action, params
        return EnvStep(
            observation=jnp.zeros((3,), dtype=jnp.float32),
            state=state,
            reward=jnp.array(0.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={},
        )


@dataclass(frozen=True)
class FakeBraxState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, object]


@dataclass(frozen=True)
class FakeBraxActuator:
    ctrl_range: jax.Array


@dataclass(frozen=True)
class FakeBraxSystem:
    actuator: FakeBraxActuator


class FakeBraxEnv:
    action_size = 2
    observation_size = 3
    sys = FakeBraxSystem(actuator=FakeBraxActuator(ctrl_range=jnp.array([[-2.0, 2.0], [0.0, 10.0]], dtype=jnp.float32)))

    def reset(self, key: chex.PRNGKey) -> FakeBraxState:
        del key
        return FakeBraxState(
            obs=jnp.zeros((3,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            info={"episode_length": jnp.array(0, dtype=jnp.int32)},
        )

    def step(self, state: FakeBraxState, action: jax.Array) -> FakeBraxState:
        del state, action
        return FakeBraxState(
            obs=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            reward=jnp.array(2.5, dtype=jnp.float32),
            done=jnp.array(False),
            info={
                "truncated": jnp.array(True),
                "contact_forces": jnp.array([0.25, 0.5], dtype=jnp.float32),
            },
        )


class TestGymnaxSpaceTranslation:
    def test_discrete_and_continuous_specs_map_to_minimal_space_objects(self):
        discrete = GymnaxCompatibilityBridge[jax.Array, jax.Array, jax.Array, None](DummyDiscreteEnv())
        continuous = GymnaxCompatibilityBridge[jax.Array, jax.Array, jax.Array, None](DummyContinuousEnv())

        observation_space = discrete.observation_space()
        discrete_action_space = discrete.action_space()
        continuous_action_space = continuous.action_space()

        assert isinstance(observation_space, GymnaxSpace)
        assert observation_space.shape == (4,)
        assert observation_space.dtype == jnp.dtype(jnp.float32)
        assert isinstance(discrete_action_space, GymnaxDiscreteSpace)
        assert discrete_action_space.shape == ()
        assert discrete_action_space.dtype == jnp.dtype(jnp.int32)
        assert discrete_action_space.n == 3
        assert isinstance(continuous_action_space, GymnaxSpace)
        assert continuous_action_space.shape == (2,)
        assert continuous_action_space.dtype == jnp.dtype(jnp.float32)


class TestGymnaxCompatibilityBridge:
    def test_step_folds_terminated_and_truncated_into_done(self):
        bridge = GymnaxCompatibilityBridge[jax.Array, jax.Array, jax.Array, None](DummyDiscreteEnv())

        observation, state = bridge.reset(jax.random.key(0))
        next_observation, next_state, reward, done, info = bridge.step(
            jax.random.key(1),
            state,
            jnp.array(1, dtype=jnp.int32),
        )

        assert observation.shape == (4,)
        assert next_observation.shape == (4,)
        assert int(next_state) == 1
        assert float(reward) == 1.0
        assert bool(done) is True
        assert bool(info["terminated"]) is False
        assert bool(info["truncated"]) is True
        assert int(info["episode_length"]) == 1

    def test_brax_adapter_step_is_gymnax_loop_compatible(self, monkeypatch):
        monkeypatch.setattr(brax_module, "_make_brax_env", lambda config: FakeBraxEnv())
        adapter = cast(EnvProtocol[jax.Array, FakeBraxState, jax.Array, None], BraxAdapter(BraxConfig(env_name="fake")))
        bridge = GymnaxCompatibilityBridge(adapter)

        spec = adapter.spec()
        observation_space = bridge.observation_space()
        action_space = bridge.action_space()
        observation, state = bridge.reset(jax.random.key(0))
        next_observation, next_state, reward, done, info = bridge.step(
            jax.random.key(1),
            state,
            jnp.array([0.1, -0.2], dtype=jnp.float32),
        )

        assert observation_space.shape == (3,)
        assert observation_space.dtype == jnp.dtype(jnp.float32)
        assert isinstance(action_space, GymnaxSpace)
        assert action_space.shape == (2,)
        assert action_space.dtype == jnp.dtype(jnp.float32)
        assert spec.action_low is not None
        assert spec.action_high is not None
        assert jnp.allclose(spec.action_low, jnp.array([-2.0, 0.0], dtype=jnp.float32))
        assert jnp.allclose(spec.action_high, jnp.array([2.0, 10.0], dtype=jnp.float32))
        assert observation.shape == (3,)
        assert next_observation.shape == (3,)
        assert next_state.obs.shape == (3,)
        assert float(reward) == 2.5
        assert bool(done) is True
        assert bool(info["terminated"]) is False
        assert bool(info["truncated"]) is True
        assert tuple(info["contact_forces"].shape) == (2,)
        assert jnp.allclose(info["contact_forces"], jnp.array([0.25, 0.5], dtype=jnp.float32))