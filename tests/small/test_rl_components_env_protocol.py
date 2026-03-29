"""Small tests for rl_components.env_protocol substrate semantics."""

import chex
import jax
import jax.numpy as jnp
import pytest
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep


class DummyEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="dummy",
            observation_shape=(84, 84, 4),
            action_shape=(),
            num_actions=4,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        observation = jnp.zeros((84, 84, 4), dtype=jnp.uint8)
        state = jnp.array(0, dtype=jnp.int32)
        return EnvReset(observation=observation, state=state)

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        observation = jnp.full((84, 84, 4), fill_value=next_state, dtype=jnp.uint8)
        return EnvStep(
            observation=observation,
            state=next_state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={"episode_length": jnp.array(1, dtype=jnp.int32)},
        )


class TestEnvSpec:
    def test_supports_discrete_and_bounded_continuous_metadata(self):
        discrete = EnvSpec(
            id="pong",
            observation_shape=(84, 84, 4),
            action_shape=(),
            observation_dtype=jnp.uint8,
            num_actions=6,
        )
        continuous = EnvSpec(
            id="ant",
            observation_shape=(27,),
            action_shape=(8,),
            action_dtype=jnp.float32,
            action_low=jnp.full((8,), -1.0, dtype=jnp.float32),
            action_high=jnp.full((8,), 1.0, dtype=jnp.float32),
        )

        assert discrete.num_actions == 6
        assert discrete.action_shape == ()
        assert continuous.num_actions is None
        assert continuous.action_shape == (8,)
        assert continuous.action_low is not None
        assert continuous.action_high is not None
        assert jnp.allclose(continuous.action_low, jnp.full((8,), -1.0, dtype=jnp.float32))
        assert jnp.allclose(continuous.action_high, jnp.full((8,), 1.0, dtype=jnp.float32))

    def test_rejects_discrete_specs_with_continuous_bounds(self):
        with pytest.raises(ValueError, match="continuous action bounds"):
            EnvSpec(
                id="broken-discrete",
                observation_shape=(4,),
                action_shape=(),
                action_dtype=jnp.int32,
                num_actions=2,
                action_low=jnp.array(-1, dtype=jnp.int32),
                action_high=jnp.array(1, dtype=jnp.int32),
            )

    def test_rejects_partial_continuous_bounds(self):
        with pytest.raises(ValueError, match="require both"):
            EnvSpec(
                id="broken-continuous",
                observation_shape=(3,),
                action_shape=(2,),
                action_dtype=jnp.float32,
                action_low=jnp.array([-1.0, -1.0], dtype=jnp.float32),
            )

    def test_allows_continuous_bounds_without_deep_semantic_validation(self):
        spec = EnvSpec(
            id="lightweight-continuous",
            observation_shape=(3,),
            action_shape=(2,),
            action_dtype=jnp.float32,
            action_low=jnp.array([0.5], dtype=jnp.float32),
            action_high=jnp.array([0.25, 1.0], dtype=jnp.float32),
        )

        assert spec.action_low is not None
        assert spec.action_high is not None
        assert spec.action_low.shape == (1,)
        assert spec.action_high.shape == (2,)

    def test_allows_traced_continuous_bounds_inside_jit(self):
        @jax.jit
        def build_spec(action_low: jax.Array, action_high: jax.Array) -> tuple[jax.Array, jax.Array]:
            spec = EnvSpec(
                id="jit-bounds",
                observation_shape=(3,),
                action_shape=(2,),
                action_dtype=jnp.float32,
                action_low=action_low,
                action_high=action_high,
            )
            assert spec.action_low is not None
            assert spec.action_high is not None
            return spec.action_low, spec.action_high

        action_low, action_high = build_spec(
            jnp.array([-1.0, 0.0], dtype=jnp.float32),
            jnp.array([1.0, 2.0], dtype=jnp.float32),
        )

        assert jnp.allclose(action_low, jnp.array([-1.0, 0.0], dtype=jnp.float32))
        assert jnp.allclose(action_high, jnp.array([1.0, 2.0], dtype=jnp.float32))


class TestEnvProtocol:
    def test_runtime_protocol_matches_expected_methods(self):
        env: EnvProtocol[jax.Array, jax.Array, jax.Array, None] = DummyEnv()
        reset = env.reset(jax.random.key(0))
        transition = env.step(jax.random.key(1), reset.state, jnp.array(2, dtype=jnp.int32))

        assert isinstance(env, EnvProtocol)
        assert reset.observation.shape == (84, 84, 4)
        assert int(transition.state) == 1
        assert float(transition.reward) == 1.0
        assert bool(transition.terminated) is False
        assert bool(transition.truncated) is False
        assert int(transition.info["episode_length"]) == 1