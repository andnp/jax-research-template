"""Small tests for rl_components.losses — QRC primal-dual saddle loss."""

import jax
import jax.numpy as jnp
from rl_components.losses import clip_gradient, qrc_loss, qrc_terms


def _apply_fn(params: dict, states: jax.Array) -> tuple[jax.Array, jax.Array]:
    v = states @ params["v"]
    h = states @ params["h"]
    return v, h


class TestQrcLoss:
    def test_zero_delta_and_dual_gives_zero_loss(self) -> None:
        params = {"v": jnp.zeros(2), "h": jnp.zeros(2)}
        states = jnp.eye(2)
        next_states = jnp.eye(2)
        rewards = jnp.zeros(2)
        discounts = jnp.zeros(2)
        loss, (delta, h) = qrc_loss(params, _apply_fn, states, next_states, rewards, discounts)
        assert jnp.allclose(loss, 0.0)
        assert jnp.allclose(delta, 0.0)
        assert jnp.allclose(h, 0.0)

    def test_delta_matches_td_target_minus_value(self) -> None:
        params = {"v": jnp.array([1.0, 0.0]), "h": jnp.zeros(2)}
        states = jnp.eye(2)
        next_states = jnp.eye(2)
        rewards = jnp.array([1.0, 1.0])
        discounts = jnp.array([0.5, 0.5])
        _, (delta, _) = qrc_loss(params, _apply_fn, states, next_states, rewards, discounts)
        # target = reward + discount * v_next; v = states @ params["v"]
        expected_target = rewards + discounts * (next_states @ params["v"])
        expected_delta = expected_target - (states @ params["v"])
        assert jnp.allclose(delta, expected_delta)

    def test_is_jittable(self) -> None:
        params = {"v": jnp.array([1.0, 0.5]), "h": jnp.array([0.1, -0.1])}
        states = jnp.eye(2)
        next_states = jnp.eye(2)
        rewards = jnp.array([0.0, 1.0])
        discounts = jnp.array([0.9, 0.9])
        def _loss(p: dict) -> jax.Array:
            return qrc_loss(p, _apply_fn, states, next_states, rewards, discounts)[0]

        loss = jax.jit(_loss)(params)
        assert bool(jnp.isfinite(loss))


def test_clip_gradient_preserves_values_and_clips_derivatives() -> None:
    """Keep forward values while clipping only oversized reverse cotangents."""
    values = jnp.array([-2.0, 0.25, 2.0])
    cotangents = jnp.array([2.0, 0.25, -3.0])

    def weighted_clip(value: jax.Array) -> jax.Array:
        return jnp.sum(clip_gradient(value, -0.5, 0.5) * cotangents)

    assert jnp.array_equal(clip_gradient(values, -0.5, 0.5), values)
    assert jnp.array_equal(
        jax.grad(weighted_clip)(values),
        jnp.array([0.5, 0.25, -0.5]),
    )


def test_delta_matches_the_hand_computed_target() -> None:
    """Verify the TD error subtracts value from the reward-plus-bootstrap target.

    The target is reward + discount * bootstrap.
    """
    q = jnp.array(0.4)
    h = jnp.array(0.0)
    reward = jnp.array(1.0)
    discount = jnp.array(0.9)
    bootstrap = jnp.array(2.0)

    _, _, delta = qrc_terms(q, h, reward, discount, bootstrap)
    assert jnp.allclose(delta, (1.0 + 0.9 * 2.0) - 0.4)


def test_h_loss_regresses_h_toward_the_stopped_delta() -> None:
    """Verify the dual loss regresses h toward the TD error.

    The target TD error is stopped before the squared residual.
    """
    q = jnp.array(0.4)
    h = jnp.array(0.1)
    reward = jnp.array(1.0)
    discount = jnp.array(0.9)
    bootstrap = jnp.array(2.0)

    _, h_loss, delta = qrc_terms(q, h, reward, discount, bootstrap)
    assert jnp.allclose(h_loss, 0.5 * (delta - h) ** 2)


def test_correction_term_gradient_flows_through_bootstrap() -> None:
    """Verify the primal correction gradient scales with discount and h.

    This gradient comes from the unstopped bootstrap in the correction term.
    """
    q, h, reward, discount = jnp.array(0.3), jnp.array(0.7), jnp.array(0.2), jnp.array(0.9)

    def v_loss_of_bootstrap(bootstrap: jax.Array) -> jax.Array:
        v_loss, _, _ = qrc_terms(q, h, reward, discount, bootstrap)
        return v_loss

    grad = jax.grad(v_loss_of_bootstrap)(jnp.array(1.5))
    assert jnp.allclose(grad, discount * h)


def test_zero_h_blocks_the_correction_gradient() -> None:
    """Verify a zero dual value leaves the primal as a semi-gradient loss.

    The bootstrap gradient is exactly zero when the correction term vanishes.
    """
    q, h, reward, discount = jnp.array(0.3), jnp.array(0.0), jnp.array(0.2), jnp.array(0.9)

    def v_loss_of_bootstrap(bootstrap: jax.Array) -> jax.Array:
        v_loss, _, _ = qrc_terms(q, h, reward, discount, bootstrap)
        return v_loss

    grad = jax.grad(v_loss_of_bootstrap)(jnp.array(1.5))
    assert jnp.allclose(grad, 0.0)
