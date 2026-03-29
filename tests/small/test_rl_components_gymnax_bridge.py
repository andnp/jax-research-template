"""Small tests for the canonical-to-Gymnax compatibility bridge."""

import chex
import jax
import jax.numpy as jnp
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
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