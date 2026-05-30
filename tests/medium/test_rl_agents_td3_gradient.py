"""Medium tests for rl_agents.td3 — gradient flow and JIT compilation."""

from dataclasses import dataclass
from typing import Any, cast

from rl_components.gym_env import ContinuousActionSpace, ObservationSpace

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_agents.td3 import Actor, Critic, TD3Config, make_train


@dataclass(frozen=True)
class FakeObservationSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32


@dataclass(frozen=True)
class FakeActionSpace:
    shape: tuple[int, ...]


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


class TestTD3GradientFlow:
    def test_make_train_accepts_injected_env(self) -> None:
        config = TD3Config(TOTAL_TIMESTEPS=4, LEARNING_STARTS=100, BUFFER_SIZE=16, BATCH_SIZE=4)
        train = make_train(config, env=FakeContinuousEnv(), env_params=None)

        out = jax.jit(train)(jax.random.key(0))
        metrics = out["metrics"]

        assert metrics["returned_episode"].shape == (4,)
        assert metrics["returned_episode_returns"].shape == (4,)

    def test_critic_params_change_after_update(self) -> None:
        critic = Critic()
        obs_dim = (4,)
        action_dim = 2
        params = critic.init(jax.random.key(0), jnp.zeros(obs_dim), jnp.zeros((action_dim,)))

        tx = optax.adam(3e-4)
        state = TrainState.create(apply_fn=critic.apply, params=params, tx=tx)

        obs = jax.random.normal(jax.random.key(1), (16, 4))
        actions = jax.random.normal(jax.random.key(2), (16, 2))
        targets = jax.random.normal(jax.random.key(3), (16,))

        def loss_fn(params: Any) -> jax.Array:
            q = cast(jax.Array, critic.apply(params, obs, actions))
            return jnp.mean(jnp.square(q - targets))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed

    def test_actor_params_change_after_update(self) -> None:
        actor = Actor(action_dim=2)
        obs_dim = (4,)
        params = actor.init(jax.random.key(0), jnp.zeros(obs_dim))

        tx = optax.adam(3e-4)
        state = TrainState.create(apply_fn=actor.apply, params=params, tx=tx)

        obs = jax.random.normal(jax.random.key(1), (16, 4))

        critic = Critic()
        critic_params = critic.init(jax.random.key(2), jnp.zeros(obs_dim), jnp.zeros((2,)))

        def loss_fn(params: Any) -> jax.Array:
            actions = cast(jax.Array, actor.apply(params, obs))
            q = cast(jax.Array, critic.apply(critic_params, obs, actions))
            return -jnp.mean(q)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed

    def test_critic_jit(self) -> None:
        critic = Critic()
        params = critic.init(jax.random.key(0), jnp.zeros((4,)), jnp.zeros((2,)))

        @jax.jit
        def forward(params: Any, obs: jax.Array, action: jax.Array) -> jax.Array:
            return critic.apply(params, obs, action)

        q = forward(params, jnp.ones((4,)), jnp.ones((2,)))
        assert q.shape == ()

    def test_actor_jit(self) -> None:
        actor = Actor(action_dim=2)
        params = actor.init(jax.random.key(0), jnp.zeros((4,)))

        @jax.jit
        def forward(params: Any, obs: jax.Array) -> jax.Array:
            return actor.apply(params, obs)

        action = forward(params, jnp.ones((4,)))
        assert action.shape == (2,)
