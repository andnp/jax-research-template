"""Small tests for rl_agents.ppo — loss function math and Transition structure."""


from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from rl_agents.ppo import (
    Transition,
    _init_observation_norm_state,
    _normalize_observation,
    _sum_action_event_terms,
    _update_observation_norm_state,
    make_train,
)
from rl_components.types import PPOConfig


class TestTransition:
    def test_is_named_tuple(self) -> None:
        t = Transition(
            done=jnp.array(0.0),
            action=jnp.array(1),
            value=jnp.array(0.5),
            reward=jnp.array(1.0),
            log_prob=jnp.array(-0.5),
            obs=jnp.zeros((4,)),
            info={},
        )
        assert isinstance(t, tuple)
        assert t[3] == 1.0


@dataclass(frozen=True)
class _ObsSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype


@dataclass(frozen=True)
class _ActionSpace:
    n: int


class _ValidationEnv:
    def observation_space(self, params: object | None = None) -> _ObsSpace:
        del params
        return _ObsSpace(shape=(4,), dtype=jnp.dtype(jnp.float32))

    def action_space(self, params: object | None = None) -> _ActionSpace:
        del params
        return _ActionSpace(n=2)

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del key, params
        return jnp.zeros((4,), dtype=jnp.float32), jnp.array(0, dtype=jnp.int32)

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, state, action, params
        info = {
            "returned_episode": jnp.array(False),
            "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
        }
        return jnp.zeros((4,), dtype=jnp.float32), jnp.array(0, dtype=jnp.int32), jnp.array(1.0), jnp.array(False), info


class TestRewardScaleValidation:
    @pytest.mark.parametrize("reward_scale", [0.0, -1.0, float("inf"), float("nan")])
    def test_make_train_rejects_non_positive_or_non_finite_reward_scale(self, reward_scale: float) -> None:
        config = PPOConfig(REWARD_SCALE=reward_scale)

        with pytest.raises(ValueError, match="REWARD_SCALE"):
            make_train(config, env=_ValidationEnv(), env_params=None)


class TestPPOClippedObjective:
    def test_no_clip_when_ratio_near_one(self) -> None:
        """When ratio ≈ 1, clipped and unclipped objectives should match."""
        ratio = jnp.array([1.0, 1.01, 0.99])
        advantages = jnp.array([1.0, -1.0, 0.5])
        clip_eps = 0.2

        loss1 = ratio * advantages
        loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        # When ratio is within [0.8, 1.2], clip has no effect
        assert jnp.allclose(loss1, loss2, atol=1e-5)

    def test_clip_limits_ratio(self) -> None:
        """Large ratio should be clipped."""
        ratio = jnp.array([2.0])
        clip_eps = 0.2

        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        assert jnp.allclose(clipped_ratio[0], 1.2)

    def test_pessimistic_bound(self) -> None:
        """PPO takes the minimum of clipped and unclipped — pessimistic bound."""
        ratio = jnp.array([1.5])
        advantages = jnp.array([1.0])
        clip_eps = 0.2

        loss1 = ratio * advantages  # 1.5
        loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages  # 1.2
        policy_loss = -jnp.minimum(loss1, loss2)  # -1.2
        assert jnp.allclose(policy_loss[0], -1.2)


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
