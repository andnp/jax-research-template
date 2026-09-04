"""Medium tests for rl_agents.td3 — gradient flow and the loss contracts.

Every test here calls the production :func:`td3_critic_loss` and
:func:`td3_actor_loss`. The assertions are differential or invariance
statements about those functions rather than a second copy of the target
arithmetic, so deleting or altering the real loss makes them fail instead of
leaving them passing against the test's own expression.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_agents.td3 import Actor, Critic, td3_actor_loss, td3_critic_loss

_OBS_DIM = 4
_ACTION_DIM = 2
_MEMBERS = 2


def _networks() -> tuple[Actor, Critic]:
    return Actor(_ACTION_DIM), Critic()


def _ensemble_params(critic: Critic, key: jax.Array) -> VariableDict:
    return jax.vmap(critic.init, in_axes=(0, None, None))(
        jax.random.split(key, _MEMBERS), jnp.zeros((_OBS_DIM,)), jnp.zeros((_ACTION_DIM,))
    )


def _batch(key: jax.Array, batch_size: int, discount: float) -> tuple[jax.Array, ...]:
    keys = jax.random.split(key, 4)
    obs = jax.random.normal(keys[0], (batch_size, _OBS_DIM))
    actions = jax.random.uniform(keys[1], (batch_size, _ACTION_DIM), minval=-1.0, maxval=1.0)
    rewards = jax.random.normal(keys[2], (batch_size,))
    next_obs = jax.random.normal(keys[3], (batch_size, _OBS_DIM))
    discounts = jnp.full((batch_size,), discount)
    return obs, actions, rewards, next_obs, discounts


def _offset_member(params: VariableDict, member: int, offset: float) -> VariableDict:
    """Shift one ensemble member's Q output by a constant, leaving the others untouched."""
    inner = dict(params["params"])
    dense = dict(inner["Dense_2"])
    dense["bias"] = dense["bias"].at[member].add(offset)
    inner["Dense_2"] = dense
    return {**params, "params": inner}


class TestTD3CriticLoss:
    def test_critic_params_change_after_update(self) -> None:
        """A gradient step on the real critic loss must move the online ensemble."""
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 16, 0.99)
        state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(3e-4))

        def loss_fn(params: VariableDict) -> jax.Array:
            return td3_critic_loss(
                params,
                actor_params,
                critic_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jax.random.key(3),
                jnp.array(0.2),
                jnp.array(0.5),
                actor=actor,
                critic=critic,
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)

        assert jnp.isfinite(loss)
        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))

    def test_terminal_transitions_ignore_next_obs(self) -> None:
        """A zero discount must drop the bootstrap, making the loss independent of ``next_obs``.

        MUTATION CHECK: dropping ``discounts *`` from ``target_q`` in
        :func:`td3_critic_loss` makes this fail, because the two wildly
        different next observations then reach the target.
        """
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, _, _ = _batch(jax.random.key(2), 8, 0.0)
        discounts = jnp.zeros((8,))

        def loss_with(next_obs: jax.Array) -> jax.Array:
            return td3_critic_loss(
                critic_params,
                actor_params,
                critic_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jax.random.key(3),
                jnp.array(0.2),
                jnp.array(0.5),
                actor=actor,
                critic=critic,
            )

        near = jax.random.normal(jax.random.key(4), (8, _OBS_DIM))
        far = jax.random.normal(jax.random.key(5), (8, _OBS_DIM)) * 100.0
        assert jnp.allclose(loss_with(near), loss_with(far))

    def test_target_takes_the_pessimistic_member(self) -> None:
        """Raising the optimistic member's values further must not move the loss.

        The target reads ``min`` over the target ensemble, so once member ``1``
        is uniformly above member ``0`` its value is inert. An aggregation that
        mixed the members in — a mean, or the maximum — would make these two
        offsets disagree.
        """
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 8, 0.99)

        def loss_with(target_params: VariableDict) -> jax.Array:
            return td3_critic_loss(
                critic_params,
                actor_params,
                target_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jax.random.key(3),
                jnp.array(0.2),
                jnp.array(0.5),
                actor=actor,
                critic=critic,
            )

        optimistic = loss_with(_offset_member(critic_params, 1, 1_000.0))
        more_optimistic = loss_with(_offset_member(critic_params, 1, 2_000.0))
        assert jnp.allclose(optimistic, more_optimistic)

    def test_critic_loss_jit(self) -> None:
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 8, 0.99)

        @jax.jit
        def compute_loss(params: VariableDict) -> jax.Array:
            return td3_critic_loss(
                params,
                actor_params,
                critic_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jax.random.key(3),
                jnp.array(0.2),
                jnp.array(0.5),
                actor=actor,
                critic=critic,
            )

        assert compute_loss(critic_params).shape == ()


class TestTD3ActorLoss:
    def test_actor_params_change_after_update(self) -> None:
        """A gradient step on the real actor loss must move the policy."""
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, _, _, _, _ = _batch(jax.random.key(2), 16, 0.99)
        state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(3e-4))

        def loss_fn(params: VariableDict) -> jax.Array:
            return td3_actor_loss(params, critic_params, obs, actor=actor, critic=critic)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)

        assert jnp.isfinite(loss)
        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))

    def test_only_the_first_member_scores_the_policy(self) -> None:
        """Perturbing member ``1`` must leave the actor loss and its gradient alone.

        The clipped minimum bounds the critic's own target; it deliberately does
        not pessimise the policy gradient, so the actor reads member ``0`` only.
        """
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, _, _, _, _ = _batch(jax.random.key(2), 8, 0.99)

        def loss_fn(params: VariableDict, critics: VariableDict) -> jax.Array:
            return td3_actor_loss(params, critics, obs, actor=actor, critic=critic)

        baseline, baseline_grads = jax.value_and_grad(loss_fn)(actor_params, critic_params)
        perturbed, perturbed_grads = jax.value_and_grad(loss_fn)(
            actor_params, _offset_member(critic_params, 1, 5.0)
        )

        assert jnp.allclose(baseline, perturbed)
        for base, pert in zip(
            jax.tree_util.tree_leaves(baseline_grads),
            jax.tree_util.tree_leaves(perturbed_grads),
            strict=True,
        ):
            assert jnp.allclose(base, pert)

    def test_actor_loss_jit(self) -> None:
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, _, _, _, _ = _batch(jax.random.key(2), 8, 0.99)

        @jax.jit
        def compute_loss(params: VariableDict) -> jax.Array:
            return td3_actor_loss(params, critic_params, obs, actor=actor, critic=critic)

        assert compute_loss(actor_params).shape == ()
