"""Medium integration tests for rl_agents.dqn with injected Nature-CNN environments."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNConfig, make_train


@dataclass(frozen=True)
class FakeObservationSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype


@dataclass(frozen=True)
class FakeActionSpace:
    n: int


class FakeAtariLikeEnv:
    def observation_space(self, params: object | None = None) -> FakeObservationSpace:
        del params
        return FakeObservationSpace(shape=(4, 84, 84, 1), dtype=jnp.uint8)

    def action_space(self, params: object | None = None) -> FakeActionSpace:
        del params
        return FakeActionSpace(n=3)

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del key, params
        observation = jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8)
        state = jnp.array(0, dtype=jnp.int32)
        return observation, state

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        observation = jnp.full((4, 84, 84, 1), next_state, dtype=jnp.uint8)
        reward = jnp.array(1.0, dtype=jnp.float32)
        done = jnp.array(False)
        info = {
            "returned_episode": jnp.array(False),
            "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
        }
        return observation, next_state, reward, done, info


class TestDQNNatureEnvIntegration:
    def test_make_train_accepts_injected_atari_like_env(self) -> None:
        config = DQNConfig(
            NETWORK_PRESET="nature_cnn",
            TOTAL_TIMESTEPS=4,
            LEARNING_STARTS=100,
            BUFFER_SIZE=16,
            BATCH_SIZE=4,
        )
        train = make_train(config, env=FakeAtariLikeEnv(), env_params=None)
        out = jax.jit(train)(jax.random.key(0))

        metrics = out["metrics"]
        assert metrics["returned_episode"].shape == (4,)
        assert metrics["returned_episode_returns"].shape == (4,)
