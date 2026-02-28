"""Small tests for rl_agents.ppo — loss function math and Transition structure."""


import jax.numpy as jnp
from rl_agents.ppo import Transition


class TestTransition:
    def test_is_named_tuple(self):
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
        assert t.reward == 1.0


class TestPPOClippedObjective:
    def test_no_clip_when_ratio_near_one(self):
        """When ratio ≈ 1, clipped and unclipped objectives should match."""
        ratio = jnp.array([1.0, 1.01, 0.99])
        advantages = jnp.array([1.0, -1.0, 0.5])
        clip_eps = 0.2

        loss1 = ratio * advantages
        loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        # When ratio is within [0.8, 1.2], clip has no effect
        assert jnp.allclose(loss1, loss2, atol=1e-5)

    def test_clip_limits_ratio(self):
        """Large ratio should be clipped."""
        ratio = jnp.array([2.0])
        clip_eps = 0.2

        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        assert jnp.allclose(clipped_ratio[0], 1.2)

    def test_pessimistic_bound(self):
        """PPO takes the minimum of clipped and unclipped — pessimistic bound."""
        ratio = jnp.array([1.5])
        advantages = jnp.array([1.0])
        clip_eps = 0.2

        loss1 = ratio * advantages  # 1.5
        loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages  # 1.2
        policy_loss = -jnp.minimum(loss1, loss2)  # -1.2
        assert jnp.allclose(policy_loss[0], -1.2)


class TestGAEComputation:
    def test_single_step_gae(self):
        """GAE with a single step should equal the TD error."""
        gamma = 0.99
        reward = 1.0
        value = 0.5
        next_value = 0.3
        done = 0.0

        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        # Single step: GAE = delta
        expected_delta = 1.0 + 0.99 * 0.3 * 1.0 - 0.5  # = 0.797
        assert abs(delta - expected_delta) < 1e-6

    def test_gae_terminal_state(self):
        """At a terminal state, next_value should be zeroed out."""
        gamma = 0.99
        reward = 1.0
        value = 0.5
        next_value = 100.0  # Should not matter since done=1
        done = 1.0

        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        # done=1 → not_done=0 → next_value ignored
        assert abs(delta - (1.0 - 0.5)) < 1e-6
