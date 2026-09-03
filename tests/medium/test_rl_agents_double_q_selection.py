"""Behavioural gate: the double-Q target must pick its bootstrap action with the *online* network.

``target = r + discount * Q_target(s', argmax_a' Q_online(s', a'))`` is the whole of what
separates ``double_dqn`` and ``dueling_dqn`` from vanilla DQN, and both agents reach it
through the one :func:`rl_agents.double_q.double_q_loss`, so a single case gates both.

The case is differential rather than arithmetic: it never recomputes the target in the test
body, which is the false confidence the deleted ``test_double_dqn_target_uses_online_for_selection``
offered. Instead it calls the real loss twice with two online parameter trees that disagree
about which next action is best and agree on everything the rest of the loss can see, so the
selector is the only free variable and the two losses must differ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.typing import VariableDict
from rl_agents import double_q
from rl_agents.double_dqn import DoubleDQNConfig
from rl_agents.q_networks import make_q_network

BATCH_SIZE = 8
OBS_DIM = 2
NUM_ACTIONS = 3
TAKEN_ACTION = 0


def _favour_action(params: VariableDict, action: int) -> VariableDict:
    """Raise one action's output bias out of reach, leaving every other action untouched.

    The offset lands on the leaf shaped like the action axis -- the output layer's bias is
    the only one -- so the surgery survives a change of hidden width, and ``action`` becomes
    the network's argmax on every row regardless of the initialisation.

    Args:
        params: Q-network parameters to shift.
        action: Index the shifted network must prefer.

    Returns:
        A new parameter tree; ``params`` is unchanged.
    """
    offset = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32).at[action].set(1e3)
    return jax.tree.map(lambda leaf: leaf + offset if leaf.shape == offset.shape else leaf, params)


def test_double_q_target_selects_the_bootstrap_action_with_the_online_network() -> None:
    """Two online trees that differ only in which next action they prefer must differ in loss."""
    network = make_q_network(DoubleDQNConfig(), NUM_ACTIONS, observation_shape=(OBS_DIM,))
    obs_zeros = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    params = network.init(jax.random.key(0), obs_zeros)
    target_params = network.init(jax.random.key(1), obs_zeros)

    obs = jax.random.normal(jax.random.key(2), (BATCH_SIZE, OBS_DIM), dtype=jnp.float32)
    actions = jnp.full((BATCH_SIZE,), TAKEN_ACTION, dtype=jnp.int32)
    rewards = jax.random.normal(jax.random.key(3), (BATCH_SIZE,), dtype=jnp.float32)
    next_obs = jax.random.normal(jax.random.key(4), (BATCH_SIZE, OBS_DIM), dtype=jnp.float32)
    discounts = jnp.full((BATCH_SIZE,), 0.99, dtype=jnp.float32)

    prefers_one = _favour_action(params, TAKEN_ACTION + 1)
    prefers_two = _favour_action(params, TAKEN_ACTION + 2)

    # Neither shift touches the taken action's column, so the predicted value the loss
    # regresses is identical under both trees and cannot explain any difference below.
    assert jnp.allclose(
        network.apply(prefers_one, obs)[:, TAKEN_ACTION],
        network.apply(prefers_two, obs)[:, TAKEN_ACTION],
    )

    def loss(online: VariableDict) -> jax.Array:
        return double_q.double_q_loss(
            online,
            target_params,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            apply_fn=network.apply,
        )

    # The mutation this defends against: selecting with the target network,
    # ``jnp.argmax(apply_fn(target_params, next_obs))`` -- equivalently vanilla DQN's
    # ``jnp.max(next_q_target, axis=-1)`` -- makes the target independent of the online
    # parameters, and both losses come out identical.
    assert not jnp.allclose(loss(prefers_one), loss(prefers_two))
