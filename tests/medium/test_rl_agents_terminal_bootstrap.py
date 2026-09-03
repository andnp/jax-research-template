"""Behavioural gate: a terminal transition must contribute no bootstrap term.

Each case calls an agent's *real* loss function with a batch whose stored ``discount`` is
exactly ``0.0`` and requires the result to be invariant to the next observation. Two wildly
different ``next_obs`` tensors are fed through the same loss; if the agent still bootstraps
across a terminal boundary the two losses differ, so the invariance formulation pins the
semantics without re-implementing the target rule or hand-computing a constant.

``rl_agents.qrc`` is already covered this way by
``test_rl_agents_qrc_gradient.py::test_terminal_transitions_zero_bootstrap_in_batch``.

``dqn`` is reachable as :func:`rl_agents.dqn.dqn_loss` and has no case here yet, so it is
gated only structurally for now, by
``test_rl_agents_learn_path.py::test_terminal_transitions_store_a_zero_discount``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from rl_agents import double_dqn, double_q, dueling_dqn, greedy_ac, replay_q_agent, sac, td3
from rl_agents.q_networks import make_q_network

BATCH_SIZE = 8
OBS_DIM = 2
ACTION_DIM = 1
NUM_ACTIONS = 3


def _next_obs_pair(shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
    """Two next-observation tensors far enough apart to separate any bootstrap term."""
    return (
        jax.random.normal(jax.random.key(10), shape, dtype=jnp.float32),
        jax.random.normal(jax.random.key(11), shape, dtype=jnp.float32) * 100.0,
    )


_DOUBLE_Q_NETWORKS: dict[str, Callable[[int], replay_q_agent.QNetworkModule]] = {
    "double_dqn": lambda action_dim: make_q_network(
        double_dqn.DoubleDQNConfig(), action_dim, observation_shape=(OBS_DIM,)
    ),
    "dueling_dqn": lambda action_dim: dueling_dqn._make_dueling_q_network(dueling_dqn.DuelingDQNConfig(), action_dim),
}
"""The network each double-Q agent binds, which is the only way the two of them differ.

They share one loss, so one case would gate the target rule. Both are listed anyway
because the network is what a port of either agent can get wrong: the second case fails if
the dueling network stops producing finite per-action values for this batch.

What it does not gate is the recombination inside the dueling head. This invariance is over
``next_obs`` alone and holds for any deterministic network, so replacing the head with a
bare linear layer satisfies it; ``tests/small/test_jax_nn_dueling_head.py`` pins the
recombination instead.
"""


@pytest.mark.parametrize("agent", list(_DOUBLE_Q_NETWORKS))
def test_double_q_loss_ignores_next_obs_on_terminal_rows(agent: str) -> None:
    """The invariance must hold only where the stored discount says it should.

    The zero-discount case is the property under test; the nonzero-discount case is its
    control. Without the control a loss that dropped its bootstrap term unconditionally --
    never bootstrapping at all -- would satisfy the invariance and look correct.
    """
    network = _DOUBLE_Q_NETWORKS[agent](NUM_ACTIONS)
    obs_zeros = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    params = network.init(jax.random.key(0), obs_zeros)
    target_params = network.init(jax.random.key(1), obs_zeros)

    obs = jax.random.normal(jax.random.key(2), (BATCH_SIZE, OBS_DIM), dtype=jnp.float32)
    actions = jax.random.randint(jax.random.key(3), (BATCH_SIZE,), 0, NUM_ACTIONS)
    rewards = jax.random.normal(jax.random.key(4), (BATCH_SIZE,), dtype=jnp.float32)
    next_obs_a, next_obs_b = _next_obs_pair((BATCH_SIZE, OBS_DIM))

    def loss(next_obs: jax.Array, discounts: jax.Array) -> jax.Array:
        return double_q.double_q_loss(
            params,
            target_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            apply_fn=network.apply,
        )

    terminal = jnp.zeros((BATCH_SIZE,), dtype=jnp.float32)
    assert jnp.allclose(loss(next_obs_a, terminal), loss(next_obs_b, terminal))

    surviving = jnp.full((BATCH_SIZE,), 0.99, dtype=jnp.float32)
    assert not jnp.allclose(loss(next_obs_a, surviving), loss(next_obs_b, surviving))


def test_sac_critic_loss_ignores_next_obs_on_terminal_rows() -> None:
    """The soft bootstrap must vanish whole: the twin-Q minimum and the entropy term alike.

    The zero-discount case is the property; the nonzero-discount case is its control,
    without which a loss that never bootstrapped at all would look correct.
    """
    actor = sac.Actor(ACTION_DIM)
    critic = sac.Critic()

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
    next_obs_a, next_obs_b = _next_obs_pair((BATCH_SIZE, OBS_DIM))

    def loss(next_obs: jax.Array, discounts: jax.Array) -> jax.Array:
        return sac.sac_critic_loss(
            critic_params,
            actor_params,
            critic_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            jnp.asarray(0.2, dtype=jnp.float32),
            jax.random.key(5),
            actor=actor,
            critic=critic,
        )

    terminal = jnp.zeros((BATCH_SIZE,), dtype=jnp.float32)
    assert jnp.allclose(loss(next_obs_a, terminal), loss(next_obs_b, terminal))

    surviving = jnp.full((BATCH_SIZE,), 0.99, dtype=jnp.float32)
    assert not jnp.allclose(loss(next_obs_a, surviving), loss(next_obs_b, surviving))


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
        return td3.td3_critic_loss(
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
