import jax
import jax.numpy as jnp
import optax
import pytest
from rl_agents.dqn_atari import (
    DQNAtariAgent,
    DQNAtariConfig,
    DQNAtariRuntimeConfig,
    dqn_atari_runtime_from_dqn_zoo,
    dqn_atari_td_loss,
    dqn_zoo_atari_exploration_decay_env_steps,
    dqn_zoo_atari_total_train_env_steps,
    dqn_zoo_atari_total_train_frames,
)
from rl_agents.dqn_zoo_atari import (
    DQNZooAtariConfig,
    build_dqn_zoo_atari_rmsprop,
    dqn_zoo_atari_frames_to_env_steps,
    dqn_zoo_atari_learn_period_env_steps,
    dqn_zoo_atari_min_replay_capacity,
    dqn_zoo_atari_target_update_period_env_steps,
)
from rl_agents.rainbow import RainbowConfig
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

        assert config.replay_capacity == 1_000_000
        assert config.min_replay_capacity_fraction == 0.05
        assert config.batch_size == 32
        assert config.num_action_repeats == 4
        assert config.target_network_update_period_frames == 40_000
        assert config.learn_period_frames == 16
        assert config.learning_rate == 0.00025
        assert config.optimizer_epsilon == 0.01 / 32**2
        assert config.rmsprop_decay == 0.95
        assert config.rmsprop_centered is True
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

    @pytest.mark.parametrize("config", [DQNAtariConfig(), RainbowConfig()], ids=["dqn_atari", "rainbow"])
    def test_rmsprop_builder_matches_literal_dqn_zoo_settings(self, config: DQNZooAtariConfig) -> None:
        """Updates equal a hand-built centered RMSProp with DQN Zoo's published constants."""
        params = jnp.array([1.0, -2.0])
        grads = jnp.array([0.5, -0.25])
        built = build_dqn_zoo_atari_rmsprop(config)
        expected = optax.rmsprop(learning_rate=0.00025, decay=0.95, eps=0.01 / 32**2, centered=True)

        built_state = built.init(params)
        expected_state = expected.init(params)
        for _ in range(3):
            built_updates, built_state = built.update(grads, built_state, params)
            expected_updates, expected_state = expected.update(grads, expected_state, params)
            assert all(
                jnp.allclose(b, e, atol=1e-7)
                for b, e in zip(jax.tree_util.tree_leaves(built_updates), jax.tree_util.tree_leaves(expected_updates), strict=True)
            )


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
        config = DQNAtariConfig(replay_capacity=8)
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
        assert state.replay_state.frame.shape == (config.replay_capacity, *OBSERVATION_SHAPE[1:])

    def test_init_rejects_a_continuous_action_space(self) -> None:
        config = DQNAtariConfig(replay_capacity=8)
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
            replay_capacity=16,
            min_replay_capacity_fraction=0.25,
            batch_size=4,
            learn_period_frames=16,
            target_network_update_period_frames=32,
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
