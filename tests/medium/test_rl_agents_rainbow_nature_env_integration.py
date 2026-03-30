from dataclasses import dataclass

import jax
import jax.numpy as jnp
from rl_agents.rainbow import RainbowConfig, make_train, rainbow_atari_runtime_from_dqn_zoo


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


class TestRainbowNatureEnvIntegration:
    def test_make_train_accepts_injected_atari_like_env(self):
        config = RainbowConfig(
            REPLAY_CAPACITY=16,
            MIN_REPLAY_CAPACITY_FRACTION=0.25,
            BATCH_SIZE=4,
            LEARN_PERIOD_FRAMES=4,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
        )
        runtime_config = rainbow_atari_runtime_from_dqn_zoo(
            config,
            num_iterations=1,
            num_train_frames_per_iteration=32,
        )
        train = make_train(config, runtime_config, env=FakeAtariLikeEnv(), env_params=None)
        out = jax.jit(train)(jax.random.key(0))

        metrics = out["metrics"]
        runner_state = out["runner_state"]
        buffer_state = runner_state[2]
        assert metrics["returned_episode"].shape == (8,)
        assert metrics["returned_episode_returns"].shape == (8,)
        assert metrics["loss"].shape == (8,)
        assert metrics["max_q"].shape == (8,)
        assert buffer_state.data["0"].dtype == jnp.uint8
        assert buffer_state.data["3"].dtype == jnp.uint8
        assert int(buffer_state.logical_capacity) == 16

    def test_make_train_inserts_n_step_transitions_and_flushes_terminals(self):
        class EpisodeCyclingEnv(FakeAtariLikeEnv):
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
                phase = (state + jnp.array(1, dtype=jnp.int32)) % jnp.array(4, dtype=jnp.int32)
                observation = jnp.full((4, 84, 84, 1), phase, dtype=jnp.uint8)
                done = phase == 0
                info = {
                    "returned_episode": done,
                    "returned_episode_returns": jnp.where(done, jnp.array(4.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
                }
                return observation, phase, jnp.array(1.0, dtype=jnp.float32), done, info

        config = RainbowConfig(
            REPLAY_CAPACITY=16,
            MIN_REPLAY_CAPACITY_FRACTION=0.25,
            BATCH_SIZE=4,
            LEARN_PERIOD_FRAMES=4,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
            N_STEP=3,
            ADDITIONAL_DISCOUNT=0.5,
        )
        runtime_config = rainbow_atari_runtime_from_dqn_zoo(
            config,
            num_iterations=1,
            num_train_frames_per_iteration=32,
        )

        train = make_train(config, runtime_config, env=EpisodeCyclingEnv(), env_params=None)
        out = jax.jit(train)(jax.random.key(0))
        buffer_state = out["runner_state"][2]

        expected_rewards = jnp.array([1.75, 1.75, 1.5, 1.0, 1.75, 1.75, 1.5, 1.0], dtype=jnp.float32)
        expected_dones = jnp.array([False, True, True, True, False, True, True, True])

        assert int(buffer_state.count) == 8
        assert jnp.array_equal(buffer_state.data["4"][:8], expected_dones)
        assert jnp.allclose(buffer_state.data["2"][:8], expected_rewards)