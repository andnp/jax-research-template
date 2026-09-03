"""Small tests for rl_agents.qrc — config validation and per-transition loss contracts."""

import jax
import jax.numpy as jnp
import pytest
from rl_agents.qrc import QRCConfig, QRCNetwork, qrc_loss


class TestQRCConfig:
    def test_defaults(self) -> None:
        cfg = QRCConfig()
        assert cfg.LR == 3e-4
        assert cfg.BUFFER_SIZE == 100_000
        assert cfg.BATCH_SIZE == 64
        assert cfg.GAMMA == 0.99
        assert cfg.EPSILON_START == 1.0
        assert cfg.EPSILON_END == 0.05
        assert cfg.BETA == 1.0

    def test_custom_config(self) -> None:
        cfg = QRCConfig(LR=1e-3, BUFFER_SIZE=1000, ENV_NAME="Acrobot-v1", BETA=0.5)
        assert cfg.LR == 1e-3
        assert cfg.BUFFER_SIZE == 1000
        assert cfg.ENV_NAME == "Acrobot-v1"
        assert cfg.BETA == 0.5

    def test_no_target_network_knobs(self) -> None:
        """QRC has no target network, so it must not carry those DQN-only knobs."""
        cfg = QRCConfig()
        assert not hasattr(cfg, "TARGET_NETWORK_FREQUENCY")
        assert not hasattr(cfg, "TAU")


class TestQRCNetwork:
    def test_output_shape(self) -> None:
        net = QRCNetwork(action_dim=4)
        params = net.init(jax.random.key(0), jnp.zeros((8,)))
        q, h = net.apply(params, jnp.ones((8,)))
        assert q.shape == (4,)
        assert h.shape == (4,)

    def test_batch_output_shape(self) -> None:
        net = QRCNetwork(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        q, h = net.apply(params, jnp.ones((10, 4)))
        assert q.shape == (10, 3)
        assert h.shape == (10, 3)

    def test_heads_are_zero_at_init(self) -> None:
        """Both heads are zero-initialised, so outputs must start at zero."""
        net = QRCNetwork(action_dim=4)
        params = net.init(jax.random.key(0), jnp.zeros((8,)))
        q, h = net.apply(params, jax.random.normal(jax.random.key(1), (8,)))
        assert jnp.allclose(q, 0.0)
        assert jnp.allclose(h, 0.0)


class TestQRCLoss:
    def test_delta_matches_bootstrap_target_with_unique_greedy_action(self) -> None:
        """With epsilon=0 and a unique argmax, delta must equal reward + gamma*max(q_next) - q[a]."""
        q = jnp.array([0.5, -0.2])
        h = jnp.zeros(2)
        q_next = jnp.array([1.0, 3.0])
        action = jnp.array(0)
        reward = jnp.array(0.5)
        gamma = jnp.array(0.9)

        _, _, delta = qrc_loss(q, h, action, reward, gamma, q_next, epsilon=0.0)

        expected_target = reward + gamma * q_next.max()
        assert jnp.allclose(delta, expected_target - q[action])

    def test_terminal_transition_ignores_next_state_value(self) -> None:
        """Passing gamma=0 (as callers do for done=True) must zero out the bootstrap."""
        q = jnp.array([0.5, -0.2])
        h = jnp.zeros(2)
        action = jnp.array(0)
        reward = jnp.array(1.0)

        _, _, delta_small_next = qrc_loss(q, h, action, reward, jnp.array(0.0), jnp.array([1.0, 2.0]), epsilon=0.1)
        _, _, delta_large_next = qrc_loss(q, h, action, reward, jnp.array(0.0), jnp.array([100.0, 200.0]), epsilon=0.1)

        assert jnp.allclose(delta_small_next, reward - q[action])
        assert jnp.allclose(delta_small_next, delta_large_next)

    def test_h_loss_gradient_vanishes_at_fixed_point(self) -> None:
        """h[a] converging to delta must zero out the h-loss gradient w.r.t. h."""
        q = jnp.array([0.5, -0.2])
        q_next = jnp.array([1.0, 3.0])
        action = jnp.array(0)
        reward = jnp.array(0.5)
        gamma = jnp.array(0.9)

        _, _, delta = qrc_loss(q, jnp.zeros(2), action, reward, gamma, q_next, epsilon=0.1)
        h_fixed_point = jnp.zeros(2).at[action].set(delta)

        def h_loss_at(h: jax.Array) -> jax.Array:
            _, h_loss, _ = qrc_loss(q, h, action, reward, gamma, q_next, epsilon=0.1)
            return h_loss

        grad_h = jax.grad(h_loss_at)(h_fixed_point)
        assert jnp.allclose(grad_h[action], 0.0, atol=1e-6)

    def test_zero_bellman_error_gives_zero_gradient(self) -> None:
        """At the true fixed point (q and h already correct), v_loss+h_loss gradients vanish."""
        q_next = jnp.array([1.0, 3.0])
        action = jnp.array(1)
        reward = jnp.array(0.5)
        gamma = jnp.array(0.9)
        epsilon = 0.1

        # Solve for the q[a] and h[a] that make delta == 0 and delta_hat == delta == 0.
        n_actions = q_next.shape[-1]
        greedy = (q_next == q_next.max()).astype(q_next.dtype)
        pi = greedy / greedy.sum()
        pi = (1.0 - epsilon) * pi + epsilon / n_actions
        v_next = q_next.dot(pi)
        target = reward + gamma * v_next

        q = jnp.zeros(2).at[action].set(target)
        h = jnp.zeros(2)

        def total_loss(q: jax.Array, h: jax.Array) -> jax.Array:
            v_loss, h_loss, _ = qrc_loss(q, h, action, reward, gamma, q_next, epsilon)
            return v_loss + h_loss

        grad_q, grad_h = jax.grad(total_loss, argnums=(0, 1))(q, h)
        assert jnp.allclose(grad_q, 0.0, atol=1e-6)
        assert jnp.allclose(grad_h, 0.0, atol=1e-6)

    def test_epsilon_greedy_mixes_uniform_over_ties(self) -> None:
        """A tie in q_next must split the greedy mass uniformly before the epsilon mix."""
        q = jnp.zeros(2)
        h = jnp.zeros(2)
        q_next = jnp.array([2.0, 2.0])  # tie
        action = jnp.array(0)
        reward = jnp.array(0.0)
        gamma = jnp.array(1.0)
        epsilon = 0.2

        _, _, delta = qrc_loss(q, h, action, reward, gamma, q_next, epsilon)

        # pi = (1-eps)*[0.5, 0.5] + eps/2 = [0.5, 0.5] regardless of eps here, v_next = 2.0
        assert jnp.allclose(delta, 2.0)


@pytest.mark.parametrize("action_dim", [1, 2, 5])
def test_qrc_network_action_dim_shapes(action_dim: int) -> None:
    net = QRCNetwork(action_dim=action_dim)
    params = net.init(jax.random.key(0), jnp.zeros((4,)))
    q, h = net.apply(params, jnp.ones((4,)))
    assert q.shape == (action_dim,)
    assert h.shape == (action_dim,)
