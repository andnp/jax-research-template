"""Small tests for rl_components.losses — QRC primal-dual saddle loss."""

import jax
import jax.numpy as jnp
from rl_components.losses import qrc_loss


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
