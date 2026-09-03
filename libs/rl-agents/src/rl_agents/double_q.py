"""The double-Q update, shared by the two agents that differ only in their network.

``double_dqn`` and ``dueling_dqn`` were identical below their network choice: the same
target, the same replay insertion, the same epsilon schedule, the same target-network
sync, down to two diverging comment lines. So ``dueling_dqn`` never implemented a dueling
variant of vanilla DQN -- it implemented *this* target with a dueling network. The update
therefore lives here once, and each agent module contributes only its public config, its
network and a three-line constructor.

The target is van Hasselt et al. (2016)::

    double-Q:      target = r + discount * Q_target(s', argmax_a' Q_online(s', a'))
    vanilla DQN:   target = r + discount * max_a' Q_target(s', a')

The online network picks the bootstrap action and the target network values it, which
splits the two roles the single ``max`` conflates and removes its overestimation bias.
That expression is this module's only contribution: everything around it -- the state
struct, ``init``, the ordering inside ``step``, the epsilon schedule, the buffer
reconstruction -- is :class:`rl_agents.replay_q_agent.ReplayQAgent`, shared with
:mod:`rl_agents.dqn`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.typing import VariableDict

from rl_agents.replay_q_agent import QApplyFn, QNetworkFactory, ReplayQAgent, ReplayQConfig


def double_q_loss(
    params: VariableDict,
    target_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    *,
    apply_fn: QApplyFn,
) -> jax.Array:
    """Mean squared error against the double-Q target, over one replay minibatch.

    Reachable from outside any agent so the termination semantics can be gated on the
    real loss rather than on a copy of it. ``discounts`` carries the bootstrap
    coefficient the loop computed, so a terminal row's ``0.0`` removes the bootstrap term
    outright and the loss becomes independent of ``next_obs`` on that row.

    Args:
        params: Online parameters. Differentiate with respect to this argument only.
        target_params: Target-network parameters, which value the selected action.
        obs: Observations the stored transitions started from.
        actions: Actions taken from ``obs``, as integer indices.
        rewards: Rewards the stored transitions earned.
        next_obs: True observations the stored transitions reached.
        discounts: Bootstrap coefficients, ``0.0`` wherever the transition terminated.
        apply_fn: The Q-network's ``apply``, static because it is a Python callable.

    Returns:
        The scalar loss.
    """
    q_values = apply_fn(params, obs)
    q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()

    next_actions = jnp.argmax(apply_fn(params, next_obs), axis=-1)
    next_q_target = apply_fn(target_params, next_obs)
    next_q_value = jnp.take_along_axis(next_q_target, next_actions[:, None], axis=-1).squeeze()

    target = rewards + discounts * next_q_value
    return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))


class DoubleQAgent(ReplayQAgent):
    """Double Q-learning with a target network and epsilon-greedy exploration.

    Not a public agent on its own: the ``network_factory`` handed to ``__init__`` decides
    which variant this is. :class:`rl_agents.double_dqn.DoubleDQNAgent` and
    :class:`rl_agents.dueling_dqn.DuelingDQNAgent` are the two bindings, and they differ
    in nothing else.
    """

    def __init__(self, config: ReplayQConfig, network_factory: QNetworkFactory) -> None:
        """Bind the configuration, the network and the double-Q target.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
            network_factory: Builds the Q-network once ``init`` knows the action count
                and the observation shape.
        """
        super().__init__(config, network_factory, double_q_loss)
