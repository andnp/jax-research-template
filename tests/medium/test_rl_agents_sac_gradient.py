"""Medium tests for rl_agents.sac — gradient flow and the three loss contracts.

The loss tests call the production :func:`sac_critic_loss`,
:func:`sac_actor_loss` and :func:`sac_alpha_loss`. Their assertions are
differential or invariance statements about those functions rather than a
second copy of the target arithmetic, so altering the real loss makes them
fail instead of leaving them passing against the test's own expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_agents.sac import (
    Actor,
    Critic,
    SACConfig,
    make_train,
    sac_actor_loss,
    sac_alpha_loss,
    sac_critic_loss,
)
from rl_components.action_normalization import ActionNormalizationWrapper
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gym_env import ContinuousActionSpace, ObservationSpace
from rl_components.gymnax_bridge import GymnaxCompatibilityBridge

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


@dataclass(frozen=True)
class FakeObservationSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32


@dataclass(frozen=True)
class FakeActionSpace:
    shape: tuple[int, ...]
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


class FakeContinuousEnv:
    def observation_space(self, params: object | None = None) -> ObservationSpace:
        del params
        return FakeObservationSpace(shape=(3,))

    def action_space(self, params: object | None = None) -> ContinuousActionSpace:
        del params
        return FakeActionSpace(shape=(2,))

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del key, params
        return jnp.zeros((3,), dtype=jnp.float32), jnp.array(0, dtype=jnp.int32)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        info = {
            "returned_episode": jnp.array(False),
            "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
        }
        return (
            jnp.full((3,), next_state, dtype=jnp.float32),
            next_state,
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array(False),
            info,
        )


class FakeCanonicalContinuousEnv:
    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="fake-canonical-continuous",
            observation_shape=(3,),
            action_shape=(2,),
            observation_dtype=jnp.float32,
            action_dtype=jnp.float32,
            action_low=jnp.array([-2.0, 0.0], dtype=jnp.float32),
            action_high=jnp.array([2.0, 4.0], dtype=jnp.float32),
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(
            observation=jnp.zeros((3,), dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        key: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        next_observation = jnp.concatenate(
            [
                jnp.asarray(action, dtype=jnp.float32),
                jnp.asarray([next_state], dtype=jnp.float32),
            ]
        )
        return EnvStep(
            observation=next_observation,
            state=next_state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={
                "returned_episode": jnp.array(False),
                "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
            },
        )


class TestSACGradientFlow:
    def test_make_train_accepts_injected_env(self) -> None:
        config = SACConfig(TOTAL_TIMESTEPS=4, LEARNING_STARTS=100, BUFFER_SIZE=16, BATCH_SIZE=4)
        train = make_train(config, env=FakeContinuousEnv(), env_params=None)

        out = jax.jit(train)(jax.random.key(0))
        metrics = out["metrics"]

        assert metrics["returned_episode"].shape == (4,)
        assert metrics["returned_episode_returns"].shape == (4,)

    def test_make_train_accepts_normalized_canonical_env_via_bridge(self) -> None:
        config = SACConfig(TOTAL_TIMESTEPS=4, LEARNING_STARTS=100, BUFFER_SIZE=16, BATCH_SIZE=4)
        canonical_env = cast(
            EnvProtocol[jax.Array, jax.Array, jax.Array, None],
            FakeCanonicalContinuousEnv(),
        )
        normalized_env = cast(
            EnvProtocol[jax.Array, jax.Array, jax.Array, None],
            ActionNormalizationWrapper(canonical_env),
        )
        gymnax_env = GymnaxCompatibilityBridge(normalized_env)

        out = jax.jit(make_train(config, env=gymnax_env, env_params=None))(jax.random.key(0))
        metrics = out["metrics"]

        assert metrics["returned_episode"].shape == (4,)
        assert metrics["returned_episode_returns"].shape == (4,)

    def test_critic_params_change_after_update(self) -> None:
        """A gradient step on the real critic loss must move the online ensemble."""
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 16, 0.99)
        state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(3e-4))

        def loss_fn(params: VariableDict) -> jax.Array:
            return sac_critic_loss(
                params,
                actor_params,
                critic_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jnp.array(0.2),
                jax.random.key(3),
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
        """A zero discount must drop the whole soft bootstrap, entropy term included.

        MUTATION CHECK: dropping ``discounts *`` from ``target_q`` in
        :func:`sac_critic_loss` makes this fail, because the two wildly
        different next observations then reach the target.
        """
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, _, _ = _batch(jax.random.key(2), 8, 0.0)
        discounts = jnp.zeros((8,))

        def loss_with(next_obs: jax.Array) -> jax.Array:
            return sac_critic_loss(
                critic_params,
                actor_params,
                critic_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jnp.array(0.2),
                jax.random.key(3),
                actor=actor,
                critic=critic,
            )

        near = jax.random.normal(jax.random.key(4), (8, _OBS_DIM))
        far = jax.random.normal(jax.random.key(5), (8, _OBS_DIM)) * 100.0
        assert jnp.allclose(loss_with(near), loss_with(far))

    def test_critic_loss_jit(self) -> None:
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 8, 0.99)

        @jax.jit
        def compute_loss(params: VariableDict) -> jax.Array:
            return sac_critic_loss(
                params,
                actor_params,
                critic_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                jnp.array(0.2),
                jax.random.key(3),
                actor=actor,
                critic=critic,
            )

        assert compute_loss(critic_params).shape == ()

    def test_actor_params_change_after_update(self) -> None:
        """A gradient step on the real actor loss must move the policy."""
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, _, _, _, _ = _batch(jax.random.key(2), 16, 0.99)
        state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(3e-4))

        def loss_fn(params: VariableDict) -> jax.Array:
            return sac_actor_loss(
                params, critic_params, obs, jnp.array(0.2), jax.random.key(3), actor=actor, critic=critic
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)

        assert jnp.isfinite(loss)
        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        assert any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))

    def test_temperature_scales_the_entropy_penalty(self) -> None:
        """Raising alpha must change the actor loss, since it prices the entropy term.

        MUTATION CHECK: dropping ``alpha *`` from the actor loss makes this
        fail, because both temperatures then produce the same value.
        """
        actor, critic = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        critic_params = _ensemble_params(critic, jax.random.key(1))
        obs, _, _, _, _ = _batch(jax.random.key(2), 8, 0.99)

        def loss_with(alpha: float) -> jax.Array:
            return sac_actor_loss(
                actor_params, critic_params, obs, jnp.array(alpha), jax.random.key(3), actor=actor, critic=critic
            )

        assert not jnp.allclose(loss_with(0.05), loss_with(5.0))

    def test_alpha_gradient_reverses_across_the_entropy_target(self) -> None:
        """log_alpha must be pushed up below the entropy target and down above it.

        Gradient descent moves ``log_alpha`` against the gradient, so a target
        entropy the policy already exceeds must yield the opposite sign to one
        it falls short of. A temperature loss that ignored ``target_entropy``
        would give the same sign for both.
        """
        actor, _ = _networks()
        actor_params = actor.init(jax.random.key(0), jnp.zeros((_OBS_DIM,)))
        alpha_params = {"log_alpha": jnp.zeros((1,), jnp.float32)}
        obs, _, _, _, _ = _batch(jax.random.key(2), 16, 0.99)

        def grad_at(target_entropy: float) -> jax.Array:
            def loss_fn(params: VariableDict) -> jax.Array:
                return sac_alpha_loss(params, actor_params, obs, target_entropy, jax.random.key(3), actor=actor)

            return jax.grad(loss_fn)(alpha_params)["log_alpha"][0]

        below_target = grad_at(10.0)
        above_target = grad_at(-10.0)
        assert below_target < 0.0 < above_target
