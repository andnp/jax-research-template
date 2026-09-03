"""Behavioural gate: a terminal transition must contribute no bootstrap term.

Each case calls an agent's *real* loss function with a batch whose stored ``discount`` is
exactly ``0.0`` and requires the result to be invariant to the next observation. Two wildly
different ``next_obs`` tensors are fed through the same loss; if the agent still bootstraps
across a terminal boundary the two losses differ, so the invariance formulation pins the
semantics without re-implementing the target rule or hand-computing a constant.

``rl_agents.qrc`` is already covered this way by
``test_rl_agents_qrc_gradient.py::test_terminal_transitions_zero_bootstrap_in_batch``.

Agents that cannot be gated here, because their loss is a closure defined inside
``make_train`` and is not reachable from outside the module: ``dqn``, ``double_dqn``,
``dueling_dqn`` and ``sac``. Exposing them belongs to each agent's own port commit, not to a
test. Until then their termination handling is gated only structurally, by
``test_rl_agents_learn_path.py::test_terminal_transitions_store_a_zero_discount``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_agents import greedy_ac, td3

BATCH_SIZE = 8
OBS_DIM = 2
ACTION_DIM = 1


def _next_obs_pair(shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
    """Two next-observation tensors far enough apart to separate any bootstrap term."""
    return (
        jax.random.normal(jax.random.key(10), shape, dtype=jnp.float32),
        jax.random.normal(jax.random.key(11), shape, dtype=jnp.float32) * 100.0,
    )


def test_td3_critic_loss_ignores_next_obs_on_terminal_rows() -> None:
    actor = td3.Actor(ACTION_DIM)
    critic = td3.Critic()
    config = td3.TD3Config()

    obs_zeros = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    action_zeros = jnp.zeros((ACTION_DIM,), dtype=jnp.float32)
    actor_params = actor.init(jax.random.key(0), obs_zeros)
    critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
        jax.random.split(jax.random.key(1), 2), obs_zeros, action_zeros
    )

    obs = jax.random.normal(jax.random.key(2), (BATCH_SIZE, OBS_DIM), dtype=jnp.float32)
    actions = jax.random.uniform(
        jax.random.key(3), (BATCH_SIZE, ACTION_DIM), dtype=jnp.float32, minval=-1.0, maxval=1.0
    )
    rewards = jax.random.normal(jax.random.key(4), (BATCH_SIZE,), dtype=jnp.float32)
    discounts = jnp.zeros((BATCH_SIZE,), dtype=jnp.float32)
    next_obs_a, next_obs_b = _next_obs_pair((BATCH_SIZE, OBS_DIM))

    def loss(next_obs: jax.Array) -> jax.Array:
        return td3._critic_loss(
            critic_params,
            actor_params,
            critic_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            jax.random.key(5),
            actor=actor,
            critic=critic,
            config=config,
        )

    assert jnp.allclose(loss(next_obs_a), loss(next_obs_b))


def test_greedy_ac_critic_loss_ignores_next_transition_on_terminal_rows() -> None:
    critic = greedy_ac.GACCritic()

    critic_params = critic.init(
        jax.random.key(0),
        jnp.zeros((OBS_DIM,), dtype=jnp.float32),
        jnp.zeros((ACTION_DIM,), dtype=jnp.float32),
    )

    obs = jax.random.normal(jax.random.key(2), (BATCH_SIZE, OBS_DIM), dtype=jnp.float32)
    actions = jax.random.uniform(
        jax.random.key(3), (BATCH_SIZE, ACTION_DIM), dtype=jnp.float32, minval=-1.0, maxval=1.0
    )
    rewards = jax.random.normal(jax.random.key(4), (BATCH_SIZE,), dtype=jnp.float32)
    discounts = jnp.zeros((BATCH_SIZE,), dtype=jnp.float32)
    next_obs_a, next_obs_b = _next_obs_pair((BATCH_SIZE, OBS_DIM))
    # The next action is the critic's other bootstrap input, so it varies with next_obs.
    next_actions_a, next_actions_b = _next_obs_pair((BATCH_SIZE, ACTION_DIM))

    def loss(next_obs: jax.Array, next_actions: jax.Array) -> jax.Array:
        total, _ = greedy_ac._batch_qrc_loss(
            critic_params, obs, actions, rewards, next_obs, discounts, next_actions, 1.0
        )
        return total

    assert jnp.allclose(loss(next_obs_a, next_actions_a), loss(next_obs_b, next_actions_b))
