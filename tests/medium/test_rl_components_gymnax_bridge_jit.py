"""Medium tests for the canonical-to-Gymnax bridge under Gymnax wrappers."""

from typing import cast

import chex
import gymnax.wrappers
import jax
import jax.numpy as jnp
import pytest
from rl_components.action_normalization import ActionNormalizationWrapper
from rl_components.brax import BraxConfig, make_brax_adapter
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gymnax_bridge import GymnaxCompatibilityBridge


class DummyCanonicalEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="dummy",
            observation_shape=(4,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
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
            reward=jnp.array(2.0, dtype=jnp.float32),
            terminated=jnp.array(True),
            truncated=jnp.array(False),
            info={"custom_metric": next_state},
        )


class TestGymnaxCompatibilityBridgeJIT:
    def test_log_wrapper_runs_over_bridge_under_jit(self):
        env = gymnax.wrappers.LogWrapper(GymnaxCompatibilityBridge[jax.Array, jax.Array, jax.Array, None](DummyCanonicalEnv()))

        reset = jax.jit(env.reset)(jax.random.key(0), None)
        observation, state = reset
        transition = jax.jit(env.step)(
            jax.random.key(1),
            state,
            jnp.array(1, dtype=jnp.int32),
            None,
        )
        next_observation, next_state, reward, done, info = transition

        assert observation.shape == (4,)
        assert next_observation.shape == (4,)
        assert int(next_state.env_state) == 1
        assert float(reward) == 2.0
        assert bool(done) is True
        assert bool(info["returned_episode"]) is True
        assert float(info["returned_episode_returns"]) == 2.0
        assert int(info["returned_episode_lengths"]) == 1
        assert int(info["custom_metric"]) == 1

    def test_real_brax_adapter_runs_through_bridge_under_jit(self):
        pytest.importorskip("brax")

        adapter = cast(EnvProtocol[jax.Array, object, jax.Array, None], make_brax_adapter(BraxConfig(env_name="inverted_pendulum")))
        bridge = GymnaxCompatibilityBridge(adapter)
        env = gymnax.wrappers.LogWrapper(bridge)
        action_space = bridge.action_space()

        observation, state = jax.jit(env.reset)(jax.random.key(0), None)
        next_observation, next_state, reward, done, info = jax.jit(env.step)(
            jax.random.key(1),
            state,
            jnp.zeros(action_space.shape, dtype=action_space.dtype),
            None,
        )

        assert observation.shape == bridge.observation_space().shape
        assert next_observation.shape == bridge.observation_space().shape
        assert reward.shape == ()
        assert done.shape == ()
        assert info["terminated"].shape == ()
        assert info["truncated"].shape == ()
        assert info["returned_episode"].shape == ()

    def test_normalized_brax_adapter_runs_through_bridge_under_jit(self):
        pytest.importorskip("brax")

        adapter = cast(EnvProtocol[jax.Array, object, jax.Array, None], make_brax_adapter(BraxConfig(env_name="inverted_pendulum")))
        normalized_adapter = cast(
            EnvProtocol[jax.Array, object, jax.Array, None],
            ActionNormalizationWrapper(adapter),
        )
        bridge = GymnaxCompatibilityBridge(normalized_adapter)
        env = gymnax.wrappers.LogWrapper(bridge)
        action_space = bridge.action_space()
        normalized_spec = normalized_adapter.spec()

        observation, state = jax.jit(env.reset)(jax.random.key(0), None)
        next_observation, next_state, reward, done, info = jax.jit(env.step)(
            jax.random.key(1),
            state,
            jnp.zeros(action_space.shape, dtype=action_space.dtype),
            None,
        )

        assert normalized_spec.action_low is not None
        assert normalized_spec.action_high is not None
        assert jnp.allclose(normalized_spec.action_low, -jnp.ones(action_space.shape, dtype=action_space.dtype))
        assert jnp.allclose(normalized_spec.action_high, jnp.ones(action_space.shape, dtype=action_space.dtype))
        assert observation.shape == bridge.observation_space().shape
        assert next_observation.shape == bridge.observation_space().shape
        assert next_state.env_state.obs.shape == bridge.observation_space().shape
        assert reward.shape == ()
        assert done.shape == ()
        assert info["terminated"].shape == ()
        assert info["truncated"].shape == ()
        assert info["returned_episode"].shape == ()