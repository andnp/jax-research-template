"""Small tests for rl_agents.ppo — loss function math and observation normalization."""


import jax.numpy as jnp
import pytest
from rl_agents.ppo import (
    PPOAgent,
    PPOConfig,
    _init_observation_norm_state,
    _normalize_observation,
    _sum_action_event_terms,
    _update_observation_norm_state,
    ppo_policy_loss,
)


class TestRewardScaleValidation:
    @pytest.mark.parametrize("reward_scale", [0.0, -1.0, float("inf"), float("nan")])
    def test_agent_rejects_non_positive_or_non_finite_reward_scale(self, reward_scale: float) -> None:
        """The scale multiplies every stored reward, so a zero or NaN must not reach a rollout."""
        with pytest.raises(ValueError, match="REWARD_SCALE"):
            PPOAgent(PPOConfig(REWARD_SCALE=reward_scale))


class TestPPOClippedObjective:
    def test_no_clip_within_the_trust_region(self) -> None:
        """Inside the trust region, the clip bound itself cannot move the loss."""
        log_prob = jnp.array([0.1, -0.2, 0.05])
        old_log_prob = jnp.array([0.09, -0.19, 0.04])
        advantages = jnp.array([1.0, -1.0, 0.5])
        valid = jnp.ones((3,), dtype=jnp.bool_)

        tight = ppo_policy_loss(log_prob, old_log_prob, advantages, valid, clip_eps=jnp.asarray(0.2))
        loose = ppo_policy_loss(log_prob, old_log_prob, advantages, valid, clip_eps=jnp.asarray(100.0))

        assert jnp.allclose(tight, loose)

    def test_ratio_growth_past_the_clip_bound_cannot_move_the_loss(self) -> None:
        """Once the ratio clears the trust region, the pessimistic bound saturates the loss.

        A positive advantage means the unclipped term grows without bound while the clipped
        term is pinned, so ``min`` selects the clipped term for any ratio past the boundary --
        further growth cannot move it.
        """
        old_log_prob = jnp.zeros((1,))
        advantages = jnp.ones((1,))
        valid = jnp.ones((1,), dtype=jnp.bool_)
        clip_eps = jnp.asarray(0.2)

        just_past = ppo_policy_loss(jnp.array([0.3]), old_log_prob, advantages, valid, clip_eps=clip_eps)
        far_past = ppo_policy_loss(jnp.array([5.0]), old_log_prob, advantages, valid, clip_eps=clip_eps)

        assert jnp.allclose(just_past, far_past)

    def test_invalid_slot_cannot_move_the_loss(self) -> None:
        """The masked mean must ignore whatever an invalid slot's ratio carries.

        Advantages are standardized across the whole minibatch by design (see
        `ppo_policy_loss`), so only the log-probabilities are perturbed here: moving the
        invalid slot's advantage would legitimately shift that standardization and make
        the assertion vacuous.
        """
        log_prob = jnp.array([0.1, -0.2])
        old_log_prob = jnp.array([0.0, 0.0])
        advantages = jnp.array([1.0, -1.0])
        valid = jnp.array([True, False])
        clip_eps = jnp.asarray(0.2)

        clean = ppo_policy_loss(log_prob, old_log_prob, advantages, valid, clip_eps=clip_eps)
        perturbed = ppo_policy_loss(
            log_prob.at[1].set(5.0),
            old_log_prob.at[1].set(-5.0),
            advantages,
            valid,
            clip_eps=clip_eps,
        )

        assert jnp.array_equal(clean, perturbed)


class TestContinuousActionReductions:
    def test_continuous_terms_reduce_last_axis(self) -> None:
        terms = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32)

        reduced = _sum_action_event_terms(terms, is_continuous=True)

        assert jnp.allclose(reduced, jnp.array([0.3, 0.7], dtype=jnp.float32))

    def test_discrete_terms_remain_unchanged(self) -> None:
        terms = jnp.array([0.1, 0.2], dtype=jnp.float32)

        reduced = _sum_action_event_terms(terms, is_continuous=False)

        assert jnp.array_equal(reduced, terms)


class TestObservationNormalization:
    def test_running_stats_track_mean_and_m2(self) -> None:
        state = _init_observation_norm_state(jnp.array([2.0, 4.0], dtype=jnp.float32))

        state = _update_observation_norm_state(state, jnp.array([4.0, 8.0], dtype=jnp.float32))

        assert jnp.allclose(state.observation_count, jnp.array(2.0, dtype=jnp.float32))
        assert jnp.allclose(state.mean, jnp.array([3.0, 6.0], dtype=jnp.float32))
        assert jnp.allclose(state.m2, jnp.array([2.0, 8.0], dtype=jnp.float32))

    def test_normalization_handles_zero_variance_without_nan(self) -> None:
        obs = jnp.array([5.0, -5.0], dtype=jnp.float32)
        state = _init_observation_norm_state(obs)

        normalized = _normalize_observation(state, obs, eps=1e-8, clip=10.0)

        assert jnp.all(jnp.isfinite(normalized))
        assert jnp.allclose(normalized, jnp.zeros_like(obs))

    def test_normalization_clips_large_values(self) -> None:
        state = _init_observation_norm_state(jnp.array([0.0], dtype=jnp.float32))

        normalized = _normalize_observation(state, jnp.array([100.0], dtype=jnp.float32), eps=1e-8, clip=3.0)

        assert jnp.allclose(normalized, jnp.array([3.0], dtype=jnp.float32))


class TestPPOConfig:
    def test_defaults(self) -> None:
        """The defaults are the published PPO hyperparameters."""
        cfg = PPOConfig()
        assert cfg.learning_rate == 2.5e-4
        assert cfg.GAE_LAMBDA == 0.95
        assert cfg.CLIP_EPS == 0.2
        assert cfg.REWARD_SCALE == 1.0
        assert cfg.ENV_NAME == "MountainCar-v0"

    def test_custom_values(self) -> None:
        """Explicit fields override their defaults."""
        cfg = PPOConfig(learning_rate=1e-3, ENV_NAME="CartPole-v1", REWARD_SCALE=2.5)
        assert cfg.learning_rate == 1e-3
        assert cfg.REWARD_SCALE == 2.5
        assert cfg.ENV_NAME == "CartPole-v1"
