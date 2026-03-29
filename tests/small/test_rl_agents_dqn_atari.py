import math

import jax
import jax.numpy as jnp
import optax
import pytest
from rl_agents.dqn_atari import (
    DQNAtariConfig,
    build_dqn_zoo_atari_rmsprop,
    dqn_zoo_atari_exploration_decay_env_steps,
    dqn_zoo_atari_exploration_epsilon,
    dqn_zoo_atari_frames_to_env_steps,
    dqn_zoo_atari_learn_period_env_steps,
    dqn_zoo_atari_min_replay_capacity,
    dqn_zoo_atari_should_learn,
    dqn_zoo_atari_target_update_period_env_steps,
    dqn_zoo_atari_total_train_env_steps,
    dqn_zoo_atari_total_train_frames,
    make_train,
)


class _ToySpace:
    def __init__(self, shape: tuple[int, ...], dtype: jnp.dtype) -> None:
        self.shape = shape
        self.dtype = dtype


class _ToyDiscreteSpace(_ToySpace):
    def __init__(self, n: int) -> None:
        super().__init__((), jnp.int32)
        self.n = n


class _ToyAtariEnv:
    def observation_space(self, params: object | None = None) -> _ToySpace:
        del params
        return _ToySpace((4, 84, 84, 1), jnp.uint8)

    def action_space(self, params: object | None = None) -> _ToyDiscreteSpace:
        del params
        return _ToyDiscreteSpace(3)

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del key, params
        return jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8), jnp.asarray(0, dtype=jnp.int32)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, action, params
        next_state = state + 1
        observation = jnp.full((4, 84, 84, 1), next_state % 255, dtype=jnp.uint8)
        returned_episode = next_state % 3 == 0
        info = {
            "returned_episode": returned_episode,
            "returned_episode_returns": jnp.where(returned_episode, jnp.asarray(1.0), jnp.asarray(0.0)),
        }
        return observation, next_state, jnp.asarray(1.0), jnp.asarray(False), info


class TestDQNAtariConfig:
    def test_defaults_match_dqn_zoo_atari_baseline(self):
        config = DQNAtariConfig()

        assert config.GAME == "pong"
        assert config.REPLAY_CAPACITY == 1_000_000
        assert config.MIN_REPLAY_CAPACITY_FRACTION == 0.05
        assert config.BATCH_SIZE == 32
        assert config.NUM_ACTION_REPEATS == 4
        assert config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES == 40_000
        assert config.LEARN_PERIOD_FRAMES == 16
        assert config.LEARNING_RATE == 0.00025
        assert config.OPTIMIZER_EPSILON == 0.01 / 32**2
        assert config.RMSPROP_DECAY == 0.95
        assert config.RMSPROP_CENTERED is True
        assert config.EXPLORATION_EPSILON_BEGIN == 1.0
        assert config.EXPLORATION_EPSILON_END == 0.1
        assert config.EXPLORATION_EPSILON_DECAY_FRAME_FRACTION == 0.02
        assert config.NUM_ITERATIONS == 200
        assert config.NUM_TRAIN_FRAMES_PER_ITERATION == 1_000_000
        assert config.EVAL_EXPLORATION_EPSILON == 0.05
        assert config.ADDITIONAL_DISCOUNT == 0.99
        assert config.SEED == 42

    def test_rmsprop_builder_uses_exact_deepmind_settings(self):
        transform = build_dqn_zoo_atari_rmsprop(DQNAtariConfig())

        assert isinstance(transform, optax.GradientTransformation)


class TestDQNAtariEnvStepConversions:
    def test_frame_counted_periods_convert_to_env_steps(self):
        config = DQNAtariConfig()

        assert dqn_zoo_atari_frames_to_env_steps(16, 4) == 4
        assert dqn_zoo_atari_learn_period_env_steps(config) == 4
        assert dqn_zoo_atari_target_update_period_env_steps(config) == 10_000
        assert dqn_zoo_atari_total_train_frames(config) == 200_000_000
        assert dqn_zoo_atari_total_train_env_steps(config) == 50_000_000
        assert dqn_zoo_atari_exploration_decay_env_steps(config) == 1_000_000

    def test_frames_to_env_steps_requires_exact_division(self):
        with pytest.raises(ValueError, match="divide evenly"):
            dqn_zoo_atari_frames_to_env_steps(17, 4)

    def test_min_replay_capacity_matches_fraction(self):
        assert dqn_zoo_atari_min_replay_capacity(DQNAtariConfig()) == 50_000


class TestDQNAtariExplorationSchedule:
    def test_epsilon_stays_at_begin_during_replay_warmup(self):
        config = DQNAtariConfig()
        warmup_last_step = dqn_zoo_atari_min_replay_capacity(config)

        assert dqn_zoo_atari_exploration_epsilon(0, config) == 1.0
        assert dqn_zoo_atari_exploration_epsilon(warmup_last_step, config) == 1.0

    def test_epsilon_decays_linearly_after_warmup_and_clamps_at_end(self):
        config = DQNAtariConfig()
        warmup = dqn_zoo_atari_min_replay_capacity(config)
        decay = dqn_zoo_atari_exploration_decay_env_steps(config)

        midpoint = warmup + decay // 2
        assert math.isclose(dqn_zoo_atari_exploration_epsilon(midpoint, config), 0.55)
        assert dqn_zoo_atari_exploration_epsilon(warmup + decay, config) == 0.1
        assert dqn_zoo_atari_exploration_epsilon(warmup + decay + 123, config) == 0.1


class TestDQNAtariLearnGating:
    def test_should_learn_requires_warmup_and_env_step_period(self):
        config = DQNAtariConfig()
        min_replay = dqn_zoo_atari_min_replay_capacity(config)

        assert dqn_zoo_atari_should_learn(49_996, min_replay - 1, config) is False
        assert dqn_zoo_atari_should_learn(50_000, min_replay, config) is True
        assert dqn_zoo_atari_should_learn(50_001, min_replay, config) is False
        assert dqn_zoo_atari_should_learn(50_004, min_replay, config) is True


class TestDQNAtariTrainPath:
    def test_make_train_runs_with_env_seam_and_emits_metrics(self):
        config = DQNAtariConfig(
            REPLAY_CAPACITY=16,
            MIN_REPLAY_CAPACITY_FRACTION=0.25,
            BATCH_SIZE=4,
            LEARN_PERIOD_FRAMES=4,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
            NUM_ITERATIONS=1,
            NUM_TRAIN_FRAMES_PER_ITERATION=32,
        )

        out = jax.jit(make_train(config, env=_ToyAtariEnv(), env_params=None))(jax.random.key(0))

        metrics = out["metrics"]
        total_env_steps = dqn_zoo_atari_total_train_env_steps(config)
        assert metrics["returned_episode"].shape == (total_env_steps,)
        assert metrics["returned_episode_returns"].shape == (total_env_steps,)
        assert metrics["epsilon"].shape == (total_env_steps,)
        assert metrics["loss"].shape == (total_env_steps,)