import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest
from rl_agents.rainbow import (
    RainbowAgent,
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
    categorical_target_support,
    rainbow_expected_q_values,
    rainbow_importance_sampling_exponent,
    rainbow_probabilities,
    rainbow_select_actions,
    rainbow_support,
)
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep


def _assert_allclose_leaf(left: jax.Array, right: jax.Array):
    npt.assert_allclose(left, right, atol=1e-7)


def _toy_atari_spec() -> EnvSpec:
    """The only thing the agent ever learns about its environment."""
    return EnvSpec(
        id="toy-atari",
        observation_shape=(4, 84, 84, 1),
        action_shape=(),
        observation_dtype=jnp.dtype(jnp.uint8),
        action_dtype=jnp.dtype(jnp.int32),
        num_actions=3,
    )


class TestRainbowConfig:
    def test_defaults_cover_c51_support_and_runtime_budget(self) -> None:
        config = RainbowConfig()
        runtime_config = RainbowRuntimeConfig()

        assert config.num_atoms == 51
        assert config.v_min == -10.0
        assert config.v_max == 10.0
        assert config.n_step == 3
        assert runtime_config.TOTAL_TRAIN_ENV_STEPS == 50_000_000

    def test_priority_exponents_match_dqn_zoo_schedule(self) -> None:
        """Expose Rainbow's configured priority alpha and beta schedule."""
        config = RainbowConfig()
        assert config.priority_exponent == 0.5
        assert jnp.allclose(rainbow_importance_sampling_exponent(jnp.asarray(0), config, 100), 0.4)
        assert jnp.allclose(rainbow_importance_sampling_exponent(jnp.asarray(50), config, 100), 0.7)
        assert jnp.allclose(rainbow_importance_sampling_exponent(jnp.asarray(100), config, 100), 1.0)
        assert jnp.allclose(rainbow_importance_sampling_exponent(jnp.asarray(200), config, 100), 1.0)

class TestRainbowNetworkAndLoss:
    def test_nature_network_emits_action_atom_logits(self) -> None:
        config = RainbowConfig()
        network = RainbowNatureNetwork(action_dim=4, num_atoms=config.num_atoms, observation_layout="fhwc")
        x = jnp.zeros((2, 4, 84, 84, 1), dtype=jnp.uint8)

        variables = network.init({"params": jax.random.key(0), "noise": jax.random.key(1)}, x)
        logits = network.apply(variables, x, rngs={"noise": jax.random.key(2)})

        assert logits.shape == (2, 4, 51)
        assert logits.dtype == jnp.float32

    def test_expected_value_action_selection_uses_distribution_mean(self) -> None:
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

    def test_categorical_target_and_loss_preserve_shapes_and_mass(self) -> None:
        support = rainbow_support(RainbowConfig(num_atoms=3, v_min=-1.0, v_max=1.0))
        next_probabilities = jnp.array(
            [
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        rewards = jnp.array([1.0, -1.0], dtype=jnp.float32)
        discounts = jnp.array([0.99, 0.0], dtype=jnp.float32)

        targets = categorical_target_probabilities(rewards, next_probabilities, support, discounts)
        loss = categorical_loss(jnp.log(jnp.clip(targets, 1e-6, 1.0)), targets)

        assert targets.shape == (2, 3)
        npt.assert_allclose(jnp.sum(targets, axis=-1), jnp.ones((2,), dtype=jnp.float32), atol=1e-6)
        assert loss.shape == ()

    def test_categorical_losses_preserve_batch_axis(self) -> None:
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

    def test_double_q_distributional_targets_use_online_selection_and_target_evaluation(self) -> None:
        support = rainbow_support(RainbowConfig(num_atoms=3, v_min=-1.0, v_max=1.0))
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
            next_probabilities=rainbow_probabilities(
                _gather_action_logits(target_next_logits, online_selected_actions)
            ),
            support=support,
            discounts=jnp.ones((2,), dtype=jnp.float32),
        )
        target_greedy_probabilities = categorical_target_probabilities(
            rewards=jnp.zeros((2,), dtype=jnp.float32),
            next_probabilities=rainbow_probabilities(
                _gather_action_logits(target_next_logits, target_greedy_actions)
            ),
            support=support,
            discounts=jnp.ones((2,), dtype=jnp.float32),
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

    def test_categorical_target_support_scales_each_row_by_its_own_discount(self) -> None:
        """One batch, three horizons: the shift must be per row, not per compiled kernel.

        Row 0 is a full-length window, row 1 ended in a termination, and row 2 is a window
        a truncation cut short. The terminal row collapses onto its own reward with no
        ``done`` mask anywhere in sight -- its stored coefficient is already ``0`` -- while
        the truncated row keeps a *different* nonzero coefficient from row 0. A single
        scalar discount cannot produce these three rows at once, which is exactly what the
        deleted ``ADDITIONAL_DISCOUNT ** N_STEP`` was.
        """
        support = rainbow_support(RainbowConfig(num_atoms=3, v_min=-1.0, v_max=1.0))

        target_support = categorical_target_support(
            rewards=jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32),
            support=support,
            discounts=jnp.array([0.25, 0.0, 0.5], dtype=jnp.float32),
        )

        npt.assert_allclose(
            target_support,
            jnp.array(
                [
                    [0.25, 0.5, 0.75],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 1.0],
                ],
                dtype=jnp.float32,
            ),
            atol=1e-6,
        )

    def test_categorical_target_probabilities_match_golden_projection_cases(self) -> None:
        support = rainbow_support(RainbowConfig(num_atoms=3, v_min=-1.0, v_max=1.0))
        n_step_projected = categorical_target_probabilities(
            rewards=jnp.array([0.5], dtype=jnp.float32),
            next_probabilities=jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
            support=support,
            discounts=jnp.array([0.25], dtype=jnp.float32),
        )
        terminal_projected = categorical_target_probabilities(
            rewards=jnp.array([0.5], dtype=jnp.float32),
            next_probabilities=jnp.array([[0.1, 0.2, 0.7]], dtype=jnp.float32),
            support=support,
            discounts=jnp.array([0.0], dtype=jnp.float32),
        )
        clipped_projected = categorical_target_probabilities(
            rewards=jnp.array([0.5], dtype=jnp.float32),
            next_probabilities=jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
            support=support,
            discounts=jnp.array([1.0], dtype=jnp.float32),
        )

        projected = jnp.concatenate(
            [
                n_step_projected,
                terminal_projected,
                clipped_projected,
            ],
            axis=0,
        )

        npt.assert_allclose(
            projected,
            jnp.array(
                [
                    [0.0, 0.25, 0.75],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
            atol=1e-6,
        )


class TestRainbowOptimizer:
    def test_agent_init_builds_rmsprop_from_config(self) -> None:
        config = RainbowConfig(
            replay_capacity=16,
            learning_rate=1e-3,
            rmsprop_decay=0.8,
            optimizer_epsilon=1e-4,
            rmsprop_centered=False,
        )

        train_state = RainbowAgent(config).init(jax.random.key(0), _toy_atari_spec()).algorithm_state.train_state
        grads = jax.tree_util.tree_map(jnp.ones_like, train_state.params)
        expected_tx = optax.rmsprop(learning_rate=1e-3, decay=0.8, eps=1e-4, centered=False)

        actual_state = train_state.tx.init(train_state.params)
        expected_state = expected_tx.init(train_state.params)
        actual_updates, actual_state = train_state.tx.update(grads, actual_state, train_state.params)
        expected_updates, expected_state = expected_tx.update(grads, expected_state, train_state.params)

        jax.tree_util.tree_map(
            _assert_allclose_leaf,
            actual_updates,
            expected_updates,
        )


class TestRainbowAgentInit:
    def test_step_metrics_and_target_schedule_follow_real_path(self) -> None:
        """Learning and target synchronization follow step-index cadence."""
        config = RainbowConfig(replay_capacity=4, batch_size=1, n_step=1, learn_period_frames=16, target_network_update_period_frames=32)
        agent = RainbowAgent(config)
        state = agent.init(jax.random.key(0), _toy_atari_spec())
        timestep = Timestep(
            reward=jnp.asarray(1.0),
            discount=jnp.asarray(0.99),
            bootstrap_observation=jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8),
            episode_end=jnp.asarray(False),
            observation=jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8),
        )
        losses = []
        targets = []
        for step_index in range(9):
            result = agent.step(state, timestep, jnp.asarray(step_index, dtype=jnp.int32))
            state = result.state
            losses.append(result.metrics["loss"])
            targets.append(state.algorithm_state.target_params)
        assert losses[3] == 0.0
        assert losses[4] != 0.0
        assert losses[5] == 0.0
        assert all(jnp.array_equal(left, right) for left, right in zip(jax.tree.leaves(targets[7]), jax.tree.leaves(targets[8]), strict=True)) is False
    def test_init_uses_per_buffer_with_logical_capacity(self) -> None:
        agent = RainbowAgent(RainbowConfig(replay_capacity=17))

        buffer_state = agent.init(jax.random.key(0), _toy_atari_spec()).replay_state.buffer_state

        assert int(buffer_state.logical_capacity) == 17
        assert int(buffer_state.storage_capacity) == 32

    def test_init_rejects_a_continuous_action_space(self) -> None:
        """The target maximizes an expected value over the actions, so there must be a count."""
        agent = RainbowAgent(RainbowConfig(replay_capacity=16))
        continuous = EnvSpec(
            id="toy-continuous",
            observation_shape=(4, 84, 84, 1),
            action_shape=(1,),
            observation_dtype=jnp.dtype(jnp.uint8),
            action_dtype=jnp.dtype(jnp.float32),
        )

        with pytest.raises(ValueError, match="discrete action space"):
            agent.init(jax.random.key(0), continuous)


def _toy_n_step_prototype() -> RainbowTransition:
    return RainbowTransition(
        obs=jnp.zeros((1,), dtype=jnp.float32),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_obs=jnp.zeros((1,), dtype=jnp.float32),
        discount=jnp.zeros((), dtype=jnp.float32),
    )


class TestRainbowNStepAccumulator:
    def test_nonterminal_insert_waits_for_exact_n_step_rewards(self) -> None:
        config = RainbowConfig(n_step=3)
        prototype = _toy_n_step_prototype()
        accumulator = _init_n_step_accumulator(config, prototype)

        for value in (1.0, 2.0):
            accumulator, _transitions, insert_mask = _advance_n_step_accumulator(
                config,
                accumulator,
                jnp.array([value], dtype=jnp.float32),
                jnp.asarray(int(value), dtype=jnp.int32),
                jnp.asarray(value, dtype=jnp.float32),
                jnp.array([value + 1.0], dtype=jnp.float32),
                jnp.asarray(0.5, dtype=jnp.float32),
                jnp.asarray(False),
            )
            assert not bool(jnp.any(insert_mask))

        accumulator, transitions, insert_mask = _advance_n_step_accumulator(
            config,
            accumulator,
            jnp.array([3.0], dtype=jnp.float32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(3.0, dtype=jnp.float32),
            jnp.array([4.0], dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.asarray(False),
        )

        assert jnp.array_equal(insert_mask, jnp.array([True, False, False]))
        npt.assert_allclose(
            transitions.reward[0],
            jnp.asarray(1.0 + 0.5 * 2.0 + 0.25 * 3.0, dtype=jnp.float32),
        )
        npt.assert_allclose(transitions.discount[0], jnp.asarray(0.125, dtype=jnp.float32))
        npt.assert_allclose(transitions.next_obs[0], jnp.array([4.0], dtype=jnp.float32))
        assert int(accumulator.size) == 2
        npt.assert_allclose(accumulator.obs[:, 0], jnp.array([2.0, 3.0, 0.0], dtype=jnp.float32))

    def test_terminal_step_flushes_all_pending_transitions(self) -> None:
        config = RainbowConfig(n_step=3)
        prototype = _toy_n_step_prototype()
        accumulator = _init_n_step_accumulator(config, prototype)

        for value in (1.0, 2.0):
            accumulator, _transitions, insert_mask = _advance_n_step_accumulator(
                config,
                accumulator,
                jnp.array([value], dtype=jnp.float32),
                jnp.asarray(int(value), dtype=jnp.int32),
                jnp.asarray(value, dtype=jnp.float32),
                jnp.array([value + 1.0], dtype=jnp.float32),
                jnp.asarray(0.5, dtype=jnp.float32),
                jnp.asarray(False),
            )
            assert not bool(jnp.any(insert_mask))

        accumulator, transitions, insert_mask = _advance_n_step_accumulator(
            config,
            accumulator,
            jnp.array([3.0], dtype=jnp.float32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(3.0, dtype=jnp.float32),
            jnp.array([4.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(True),
        )

        assert jnp.array_equal(insert_mask, jnp.array([True, True, True]))
        npt.assert_allclose(transitions.reward, jnp.array([2.75, 3.5, 3.0], dtype=jnp.float32))
        npt.assert_allclose(transitions.discount, jnp.zeros((3,), dtype=jnp.float32))
        npt.assert_allclose(transitions.next_obs[:, 0], jnp.array([4.0, 4.0, 4.0], dtype=jnp.float32))
        assert int(accumulator.size) == 0

    def test_truncation_gives_every_flushed_row_its_own_horizon_discount(self) -> None:
        """A boundary that keeps its bootstrap flushes the window at three horizons at once.

        Termination is the easy half: ``discount = 0`` erases the tail of the return by
        itself, and one shared value is right for every flushed row. Truncation is the
        half a boolean cannot encode -- the trajectory breaks while ``discount = gamma``
        survives -- so the row flushed at offset ``i`` bootstraps over ``m - i`` steps and
        must leave with ``gamma ** (m - i)``. A single ``gamma ** N_STEP`` is correct for
        at most the oldest row, and the assertion below pins all three apart.
        """
        config = RainbowConfig(n_step=3)
        accumulator = _init_n_step_accumulator(config, _toy_n_step_prototype())
        gamma = jnp.asarray(0.5, dtype=jnp.float32)

        for value in (1.0, 2.0):
            accumulator, _transitions, insert_mask = _advance_n_step_accumulator(
                config,
                accumulator,
                jnp.array([value], dtype=jnp.float32),
                jnp.asarray(int(value), dtype=jnp.int32),
                jnp.asarray(value, dtype=jnp.float32),
                jnp.array([value + 1.0], dtype=jnp.float32),
                gamma,
                jnp.asarray(False),
            )
            assert not bool(jnp.any(insert_mask))

        accumulator, transitions, insert_mask = _advance_n_step_accumulator(
            config,
            accumulator,
            jnp.array([3.0], dtype=jnp.float32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(3.0, dtype=jnp.float32),
            jnp.array([4.0], dtype=jnp.float32),
            gamma,
            jnp.asarray(True),
        )

        assert jnp.array_equal(insert_mask, jnp.array([True, True, True]))
        npt.assert_allclose(transitions.discount, jnp.array([0.125, 0.25, 0.5], dtype=jnp.float32))
        assert not bool(jnp.all(transitions.discount == transitions.discount[0])), (
            "a truncation flushed three horizons with one shared discount"
        )
        npt.assert_allclose(transitions.reward, jnp.array([2.75, 3.5, 3.0], dtype=jnp.float32))
        npt.assert_allclose(transitions.next_obs[:, 0], jnp.array([4.0, 4.0, 4.0], dtype=jnp.float32))
        assert int(accumulator.size) == 0
