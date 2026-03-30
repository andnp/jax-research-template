from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from rl_agents.dqn_atari import (
    DQNAtariConfig,
    dqn_zoo_atari_frames_to_env_steps,
    dqn_zoo_atari_should_learn,
)
from rl_agents.rainbow import (
    RainbowConfig,
    RainbowNatureNetwork,
    RainbowRuntimeConfig,
    RainbowTransition,
    _advance_n_step_accumulator,
    _gather_action_logits,
    _init_n_step_accumulator,
    categorical_loss,
    categorical_losses,
    categorical_target_probabilities,
    initialize_train_state,
    make_train,
    rainbow_atari_runtime_from_dqn_zoo,
    rainbow_expected_q_values,
    rainbow_probabilities,
    rainbow_select_actions,
    rainbow_support,
    rainbow_zoo_atari_frames_to_env_steps,
    rainbow_zoo_atari_should_learn,
    rainbow_zoo_atari_should_update_target,
    rainbow_zoo_atari_total_train_env_steps,
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
        info = {
            "returned_episode": next_state % 3 == 0,
            "returned_episode_returns": jnp.where(next_state % 3 == 0, jnp.asarray(1.0), jnp.asarray(0.0)),
        }
        return observation, next_state, jnp.asarray(1.0), jnp.asarray(False), info


class TestRainbowConfig:
    def test_defaults_cover_c51_support_and_runtime_budget(self):
        config = RainbowConfig()
        runtime_config = RainbowRuntimeConfig()

        assert config.NUM_ATOMS == 51
        assert config.V_MIN == -10.0
        assert config.V_MAX == 10.0
        assert config.N_STEP == 3
        assert runtime_config.TOTAL_TRAIN_ENV_STEPS == 50_000_000

    def test_runtime_helper_preserves_dqn_zoo_budget(self):
        config = RainbowConfig()
        runtime_config = rainbow_atari_runtime_from_dqn_zoo(config)

        assert runtime_config.TOTAL_TRAIN_ENV_STEPS == 50_000_000
        assert rainbow_zoo_atari_total_train_env_steps(runtime_config) == 50_000_000

    def test_schedule_helpers_match_dqn_atari_frame_conversion(self):
        rainbow_config = RainbowConfig()
        dqn_config = DQNAtariConfig()

        assert rainbow_zoo_atari_frames_to_env_steps(
            rainbow_config.LEARN_PERIOD_FRAMES,
            rainbow_config.NUM_ACTION_REPEATS,
        ) == dqn_zoo_atari_frames_to_env_steps(
            dqn_config.LEARN_PERIOD_FRAMES,
            dqn_config.NUM_ACTION_REPEATS,
        )
        assert rainbow_zoo_atari_frames_to_env_steps(
            rainbow_config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES,
            rainbow_config.NUM_ACTION_REPEATS,
        ) == dqn_zoo_atari_frames_to_env_steps(
            dqn_config.TARGET_NETWORK_UPDATE_PERIOD_FRAMES,
            dqn_config.NUM_ACTION_REPEATS,
        )

    @pytest.mark.parametrize(
        ("env_step", "replay_size"),
        [
            (0, 0),
            (3, 50_000),
            (4, 49_999),
            (4, 50_000),
            (8, 50_000),
        ],
    )
    def test_should_learn_matches_dqn_atari_warmup_and_cadence(self, env_step: int, replay_size: int):
        rainbow_config = RainbowConfig()
        dqn_config = DQNAtariConfig()

        assert rainbow_zoo_atari_should_learn(env_step, replay_size, rainbow_config) is dqn_zoo_atari_should_learn(
            env_step,
            replay_size,
            dqn_config,
        )

    @pytest.mark.parametrize(
        ("env_step", "expected"),
        [
            (0, True),
            (9_999, False),
            (10_000, True),
            (10_001, False),
        ],
    )
    def test_should_update_target_uses_env_step_cadence(self, env_step: int, expected: bool):
        config = RainbowConfig()

        assert rainbow_zoo_atari_should_update_target(env_step, config) is expected


class TestRainbowNetworkAndLoss:
    def test_nature_network_emits_action_atom_logits(self):
        config = RainbowConfig()
        network = RainbowNatureNetwork(action_dim=4, num_atoms=config.NUM_ATOMS, observation_layout="fhwc")
        x = jnp.zeros((2, 4, 84, 84, 1), dtype=jnp.uint8)

        variables = network.init({"params": jax.random.key(0), "noise": jax.random.key(1)}, x)
        logits = network.apply(variables, x, rngs={"noise": jax.random.key(2)})

        assert logits.shape == (2, 4, 51)
        assert logits.dtype == jnp.float32

    def test_expected_value_action_selection_uses_distribution_mean(self):
        support = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
        logits = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [-5.0, -5.0, 5.0],
            ],
            dtype=jnp.float32,
        )

        q_values = rainbow_expected_q_values(logits, support)
        action = rainbow_select_actions(logits, support)

        assert q_values.shape == (2,)
        assert int(action) == 1

    def test_categorical_target_and_loss_preserve_shapes_and_mass(self):
        support = rainbow_support(RainbowConfig(NUM_ATOMS=3, V_MIN=-1.0, V_MAX=1.0))
        next_probabilities = jnp.array(
            [
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        rewards = jnp.array([1.0, -1.0], dtype=jnp.float32)
        dones = jnp.array([0.0, 1.0], dtype=jnp.float32)

        targets = categorical_target_probabilities(rewards, dones, next_probabilities, support, 0.99)
        loss = categorical_loss(jnp.log(jnp.clip(targets, 1e-6, 1.0)), targets)

        assert targets.shape == (2, 3)
        npt.assert_allclose(jnp.sum(targets, axis=-1), jnp.ones((2,), dtype=jnp.float32), atol=1e-6)
        assert loss.shape == ()

    def test_categorical_losses_preserve_batch_axis(self):
        logits = jnp.log(
            jnp.array(
                [
                    [0.2, 0.5, 0.3],
                    [0.1, 0.6, 0.3],
                ],
                dtype=jnp.float32,
            )
        )
        targets = jnp.array(
            [
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        losses = categorical_losses(logits, targets)

        assert losses.shape == (2,)
        assert jnp.all(losses >= 0.0)

    def test_double_q_distributional_targets_use_online_selection_and_target_evaluation(self):
        support = rainbow_support(RainbowConfig(NUM_ATOMS=3, V_MIN=-1.0, V_MAX=1.0))
        online_next_probabilities = jnp.array(
            [
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.2, 0.7],
                ],
                [
                    [0.2, 0.2, 0.6],
                    [0.6, 0.2, 0.2],
                ],
            ],
            dtype=jnp.float32,
        )
        target_next_probabilities = jnp.array(
            [
                [
                    [0.1, 0.1, 0.8],
                    [0.7, 0.2, 0.1],
                ],
                [
                    [0.6, 0.2, 0.2],
                    [0.1, 0.2, 0.7],
                ],
            ],
            dtype=jnp.float32,
        )
        online_next_logits = jnp.log(online_next_probabilities)
        target_next_logits = jnp.log(target_next_probabilities)

        online_selected_actions = rainbow_select_actions(online_next_logits, support)
        target_greedy_actions = rainbow_select_actions(target_next_logits, support)

        assert jnp.array_equal(online_selected_actions, jnp.array([1, 0], dtype=jnp.int32))
        assert jnp.array_equal(target_greedy_actions, jnp.array([0, 1], dtype=jnp.int32))

        double_q_target_probabilities = categorical_target_probabilities(
            rewards=jnp.zeros((2,), dtype=jnp.float32),
            dones=jnp.zeros((2,), dtype=jnp.float32),
            next_probabilities=rainbow_probabilities(
                _gather_action_logits(target_next_logits, online_selected_actions)
            ),
            support=support,
            discount=1.0,
        )
        target_greedy_probabilities = categorical_target_probabilities(
            rewards=jnp.zeros((2,), dtype=jnp.float32),
            dones=jnp.zeros((2,), dtype=jnp.float32),
            next_probabilities=rainbow_probabilities(
                _gather_action_logits(target_next_logits, target_greedy_actions)
            ),
            support=support,
            discount=1.0,
        )

        npt.assert_allclose(
            double_q_target_probabilities,
            jnp.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.6, 0.2, 0.2],
                ],
                dtype=jnp.float32,
            ),
            atol=1e-6,
        )
        assert not jnp.allclose(double_q_target_probabilities, target_greedy_probabilities)


class TestRainbowTrainPath:
    def test_make_train_requires_explicit_env(self):
        train_factory = cast(Callable[..., object], make_train)

        with pytest.raises(TypeError):
            train_factory(RainbowConfig(), RainbowRuntimeConfig())

    def test_initialize_train_state_uses_per_buffer_with_logical_capacity(self):
        config = RainbowConfig(REPLAY_CAPACITY=1_000_000)
        _network, _prototype, runner_state = initialize_train_state(
            config,
            _ToyAtariEnv(),
            jax.random.key(0),
        )

        buffer_state = runner_state[2]

        assert int(buffer_state.logical_capacity) == 1_000_000
        assert int(buffer_state.storage_capacity) == 1_048_576


class TestRainbowNStepAccumulator:
    def test_nonterminal_insert_waits_for_exact_n_step_rewards(self):
        config = RainbowConfig(N_STEP=3, ADDITIONAL_DISCOUNT=0.5)
        prototype = RainbowTransition(
            obs=jnp.zeros((1,), dtype=jnp.float32),
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros((), dtype=jnp.float32),
            next_obs=jnp.zeros((1,), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.bool_),
        )
        accumulator = _init_n_step_accumulator(config, prototype)

        accumulator, _transitions, insert_mask = _advance_n_step_accumulator(
            config,
            prototype,
            accumulator,
            jnp.array([1.0], dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
            jnp.asarray(False),
        )
        assert not bool(jnp.any(insert_mask))

        accumulator, _transitions, insert_mask = _advance_n_step_accumulator(
            config,
            prototype,
            accumulator,
            jnp.array([2.0], dtype=jnp.float32),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(2.0, dtype=jnp.float32),
            jnp.array([3.0], dtype=jnp.float32),
            jnp.asarray(False),
        )
        assert not bool(jnp.any(insert_mask))

        accumulator, transitions, insert_mask = _advance_n_step_accumulator(
            config,
            prototype,
            accumulator,
            jnp.array([3.0], dtype=jnp.float32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(3.0, dtype=jnp.float32),
            jnp.array([4.0], dtype=jnp.float32),
            jnp.asarray(False),
        )

        assert jnp.array_equal(insert_mask, jnp.array([True, False, False]))
        npt.assert_allclose(
            transitions.reward[0],
            jnp.asarray(1.0 + 0.5 * 2.0 + 0.25 * 3.0, dtype=jnp.float32),
        )
        assert bool(transitions.done[0]) is False
        npt.assert_allclose(transitions.next_obs[0], jnp.array([4.0], dtype=jnp.float32))
        assert int(accumulator.size) == 2
        npt.assert_allclose(accumulator.obs[:, 0], jnp.array([2.0, 3.0, 0.0], dtype=jnp.float32))

    def test_terminal_step_flushes_all_pending_transitions(self):
        config = RainbowConfig(N_STEP=3, ADDITIONAL_DISCOUNT=0.5)
        prototype = RainbowTransition(
            obs=jnp.zeros((1,), dtype=jnp.float32),
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros((), dtype=jnp.float32),
            next_obs=jnp.zeros((1,), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.bool_),
        )
        accumulator = _init_n_step_accumulator(config, prototype)

        for value in (1.0, 2.0):
            accumulator, _transitions, insert_mask = _advance_n_step_accumulator(
                config,
                prototype,
                accumulator,
                jnp.array([value], dtype=jnp.float32),
                jnp.asarray(int(value), dtype=jnp.int32),
                jnp.asarray(value, dtype=jnp.float32),
                jnp.array([value + 1.0], dtype=jnp.float32),
                jnp.asarray(False),
            )
            assert not bool(jnp.any(insert_mask))

        accumulator, transitions, insert_mask = _advance_n_step_accumulator(
            config,
            prototype,
            accumulator,
            jnp.array([3.0], dtype=jnp.float32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(3.0, dtype=jnp.float32),
            jnp.array([4.0], dtype=jnp.float32),
            jnp.asarray(True),
        )

        assert jnp.array_equal(insert_mask, jnp.array([True, True, True]))
        npt.assert_allclose(transitions.reward, jnp.array([2.75, 3.5, 3.0], dtype=jnp.float32))
        assert jnp.array_equal(transitions.done, jnp.array([True, True, True]))
        npt.assert_allclose(transitions.next_obs[:, 0], jnp.array([4.0, 4.0, 4.0], dtype=jnp.float32))
        assert int(accumulator.size) == 0
