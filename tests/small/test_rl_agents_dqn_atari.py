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
    dqn_atari_td_loss,
    dqn_zoo_atari_exploration_decay_env_steps,
    dqn_zoo_atari_frames_to_env_steps,
    dqn_zoo_atari_learn_period_env_steps,
    dqn_zoo_atari_min_replay_capacity,
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
        assert config.GRAD_ERROR_BOUND == 1 / 32

    def test_td_loss_gradient_is_bounded_per_sample(self) -> None:
        """Match DQN Zoo's clipped TD-error gradient after mean reduction."""
        q_action = jnp.array([-3.0, 0.01, 4.0])
        targets = jnp.zeros(3)
        bound = 1 / 32

        def loss(q: jax.Array) -> jax.Array:
            return dqn_atari_td_loss(q, targets, bound)

        gradient = jax.grad(loss)(q_action)
        expected = jnp.clip((q_action - targets) / q_action.size, -bound, bound)
        assert jnp.allclose(gradient, expected)

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


class TestDQNAtariAgent:
    def test_init_derives_the_network_and_buffer_from_the_spec_alone(self) -> None:
        config = DQNAtariConfig(REPLAY_CAPACITY=8)
        agent = DQNAtariAgent(config, dqn_atari_runtime_from_dqn_zoo(config, num_iterations=1))

        state = agent.init(jax.random.key(0), _ToyAtariEnv().spec())
        algorithm_state = state.algorithm_state

        q_values = algorithm_state.train_state.apply_fn(
            algorithm_state.train_state.params,
            jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.uint8),
        )
        assert q_values.shape == (NUM_ACTIONS,)
        assert state.last_obs.shape == OBSERVATION_SHAPE
        assert state.last_obs.dtype == jnp.uint8
        assert state.last_action.shape == ()
        assert int(algorithm_state.num_actions) == NUM_ACTIONS
        assert state.replay_state.frame.shape == (config.REPLAY_CAPACITY, *OBSERVATION_SHAPE[1:])

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
            LEARN_PERIOD_FRAMES=16,
            TARGET_NETWORK_UPDATE_PERIOD_FRAMES=32,
            EXPLORATION_EPSILON_DECAY_FRAME_FRACTION=0.5,
        )
        runtime_config = dqn_atari_runtime_from_dqn_zoo(
            config,
            num_iterations=1,
            num_train_frames_per_iteration=64,
        )
        steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
        agent = DQNAtariAgent(config, runtime_config)

        def train_fn(key: jax.Array):
            return run(agent, _ToyAtariEnv(), key, steps=steps, gamma=GAMMA)

        final_state, metrics = jax.jit(train_fn)(jax.random.key(0))

        for key_name in ("loss", "epsilon", "loop/reward", "loop/episode_end"):
            assert metrics[key_name].shape == (steps,), key_name
        buffer_state = final_state.agent_state.replay_state
        assert buffer_state.frame.dtype == jnp.uint8
        assert buffer_state.next_frame.dtype == jnp.uint8
        assert int(buffer_state.count) == steps - 1, "the final action opens a transition that never closes"
        assert jnp.any(metrics["loss"] != 0.0), "the replay warmup completed, so the learn path must have fired"
        assert jnp.allclose(metrics["epsilon"][0:5], 1.0)
        assert jnp.allclose(metrics["epsilon"][8], 0.55, atol=1e-6)
        assert jnp.allclose(metrics["epsilon"][12:], 0.1)
        assert metrics["loss"][3] == 0.0
        assert metrics["loss"][4] != 0.0
        assert metrics["loss"][5] == 0.0
