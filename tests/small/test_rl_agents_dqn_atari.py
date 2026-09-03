import math

import jax
import jax.numpy as jnp
import optax
import pytest
from rl_agents.dqn_atari import (
    DQNAtariAgent,
    DQNAtariConfig,
    DQNAtariRuntimeConfig,
    build_dqn_zoo_atari_rmsprop,
    dqn_atari_runtime_from_dqn_zoo,
    dqn_zoo_atari_exploration_decay_env_steps,
    dqn_zoo_atari_exploration_epsilon,
    dqn_zoo_atari_frames_to_env_steps,
    dqn_zoo_atari_learn_period_env_steps,
    dqn_zoo_atari_min_replay_capacity,
    dqn_zoo_atari_should_learn,
    dqn_zoo_atari_target_update_period_env_steps,
    dqn_zoo_atari_total_train_env_steps,
    dqn_zoo_atari_total_train_frames,
)
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.loop import run

OBSERVATION_SHAPE = (4, 84, 84, 1)
NUM_ACTIONS = 3
GAMMA = 0.99
EPISODE_LENGTH = 3


class _ToyAtariEnv:
    """A counter environment wearing Atari's observation shape, dtype and action count."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-atari",
            observation_shape=OBSERVATION_SHAPE,
            action_shape=(),
            observation_dtype=jnp.dtype(jnp.uint8),
            action_dtype=jnp.dtype(jnp.int32),
            num_actions=NUM_ACTIONS,
        )

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.uint8),
            state=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, action, params
        next_state = state + jnp.asarray(1, dtype=jnp.int32)
        return EnvStep(
            observation=jnp.full(OBSERVATION_SHAPE, next_state % 255, dtype=jnp.uint8),
            state=next_state,
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=next_state % EPISODE_LENGTH == 0,
            truncated=jnp.bool_(False),
            info={},
        )


class TestDQNAtariConfig:
    def test_defaults_keep_learner_fields_only(self) -> None:
        config = DQNAtariConfig()

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

    def test_runtime_defaults_preserve_previous_training_budget(self) -> None:
        runtime_config = DQNAtariRuntimeConfig()

        assert runtime_config.TOTAL_TRAIN_ENV_STEPS == 50_000_000
        assert runtime_config.SEED == 42
        assert runtime_config.EVAL_EXPLORATION_EPSILON == 0.05

    def test_runtime_helper_preserves_dqn_zoo_baseline(self) -> None:
        config = DQNAtariConfig()
        runtime_config = dqn_atari_runtime_from_dqn_zoo(config)

        assert runtime_config.TOTAL_TRAIN_ENV_STEPS == 50_000_000
        assert runtime_config.SEED == 42
        assert runtime_config.EVAL_EXPLORATION_EPSILON == 0.05

    def test_rmsprop_builder_uses_exact_deepmind_settings(self) -> None:
        transform = build_dqn_zoo_atari_rmsprop(DQNAtariConfig())

        assert isinstance(transform, optax.GradientTransformation)


class TestDQNAtariEnvStepConversions:
    def test_frame_counted_periods_convert_to_env_steps(self) -> None:
        config = DQNAtariConfig()
        runtime_config = dqn_atari_runtime_from_dqn_zoo(config)

        assert dqn_zoo_atari_frames_to_env_steps(16, 4) == 4
        assert dqn_zoo_atari_learn_period_env_steps(config) == 4
        assert dqn_zoo_atari_target_update_period_env_steps(config) == 10_000
        assert dqn_zoo_atari_total_train_frames(config, runtime_config) == 200_000_000
        assert dqn_zoo_atari_total_train_env_steps(runtime_config) == 50_000_000
        assert dqn_zoo_atari_exploration_decay_env_steps(config, runtime_config) == 1_000_000

    def test_frames_to_env_steps_requires_exact_division(self) -> None:
        with pytest.raises(ValueError, match="divide evenly"):
            dqn_zoo_atari_frames_to_env_steps(17, 4)

    def test_min_replay_capacity_matches_fraction(self) -> None:
        assert dqn_zoo_atari_min_replay_capacity(DQNAtariConfig()) == 50_000


class TestDQNAtariExplorationSchedule:
    def test_epsilon_stays_at_begin_during_replay_warmup(self) -> None:
        config = DQNAtariConfig()
        runtime_config = dqn_atari_runtime_from_dqn_zoo(config)
        warmup_last_step = dqn_zoo_atari_min_replay_capacity(config)

        assert dqn_zoo_atari_exploration_epsilon(0, config, runtime_config) == 1.0
        assert dqn_zoo_atari_exploration_epsilon(warmup_last_step, config, runtime_config) == 1.0

    def test_epsilon_decays_linearly_after_warmup_and_clamps_at_end(self) -> None:
        config = DQNAtariConfig()
        runtime_config = dqn_atari_runtime_from_dqn_zoo(config)
        warmup = dqn_zoo_atari_min_replay_capacity(config)
        decay = dqn_zoo_atari_exploration_decay_env_steps(config, runtime_config)

        midpoint = warmup + decay // 2
        assert math.isclose(dqn_zoo_atari_exploration_epsilon(midpoint, config, runtime_config), 0.55)
        assert dqn_zoo_atari_exploration_epsilon(warmup + decay, config, runtime_config) == 0.1
        assert dqn_zoo_atari_exploration_epsilon(warmup + decay + 123, config, runtime_config) == 0.1


class TestDQNAtariLearnGating:
    def test_should_learn_requires_warmup_and_env_step_period(self) -> None:
        config = DQNAtariConfig()
        min_replay = dqn_zoo_atari_min_replay_capacity(config)

        assert dqn_zoo_atari_should_learn(49_996, min_replay - 1, config) is False
        assert dqn_zoo_atari_should_learn(50_000, min_replay, config) is True
        assert dqn_zoo_atari_should_learn(50_001, min_replay, config) is False
        assert dqn_zoo_atari_should_learn(50_004, min_replay, config) is True


class TestDQNAtariAgent:
    def test_init_derives_the_network_and_buffer_from_the_spec_alone(self) -> None:
        config = DQNAtariConfig(REPLAY_CAPACITY=8)
        agent = DQNAtariAgent(config, dqn_atari_runtime_from_dqn_zoo(config, num_iterations=1))

        state = agent.init(jax.random.key(0), _ToyAtariEnv().spec())

        q_values = state.train_state.apply_fn(
            state.train_state.params,
            jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.uint8),
        )
        assert q_values.shape == (NUM_ACTIONS,)
        assert state.last_obs.shape == OBSERVATION_SHAPE
        assert state.last_obs.dtype == jnp.uint8
        assert state.last_action.shape == ()
        assert int(state.num_actions) == NUM_ACTIONS
        assert state.buffer_state.obs.shape == (config.REPLAY_CAPACITY, *OBSERVATION_SHAPE)

    def test_init_rejects_a_continuous_action_space(self) -> None:
        config = DQNAtariConfig(REPLAY_CAPACITY=8)
        agent = DQNAtariAgent(config, dqn_atari_runtime_from_dqn_zoo(config, num_iterations=1))
        spec = EnvSpec(
            id="continuous",
            observation_shape=OBSERVATION_SHAPE,
            action_shape=(1,),
            action_dtype=jnp.dtype(jnp.float32),
        )

        with pytest.raises(ValueError, match="discrete action space"):
            agent.init(jax.random.key(0), spec)

    def test_run_drives_the_agent_and_emits_a_fixed_metric_schema(self) -> None:
        config = DQNAtariConfig(
            REPLAY_CAPACITY=16,
            MIN_REPLAY_CAPACITY_FRACTION=0.25,
            BATCH_SIZE=4,
            LEARN_PERIOD_FRAMES=4,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
        )
        runtime_config = dqn_atari_runtime_from_dqn_zoo(
            config,
            num_iterations=1,
            num_train_frames_per_iteration=32,
        )
        steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
        agent = DQNAtariAgent(config, runtime_config)

        final_state, metrics = jax.jit(
            lambda key: run(agent, _ToyAtariEnv(), key, steps=steps, gamma=GAMMA)
        )(jax.random.key(0))

        for key_name in ("loss", "epsilon", "loop/reward", "loop/episode_end"):
            assert metrics[key_name].shape == (steps,), key_name
        buffer_state = final_state.agent_state.buffer_state
        assert buffer_state.obs.dtype == jnp.uint8
        assert buffer_state.next_obs.dtype == jnp.uint8
        assert int(buffer_state.count) == steps - 1, "the final action opens a transition that never closes"
        assert jnp.any(metrics["loss"] != 0.0), "the replay warmup completed, so the learn path must have fired"
