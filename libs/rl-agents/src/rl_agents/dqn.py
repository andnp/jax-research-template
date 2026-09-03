"""DQN as an :class:`~rl_components.agent_protocol.AgentProtocol` implementation.

Everything below the target rule -- the state struct, ``init``, the normative
insert/learn/sync/act ordering in ``step``, the epsilon schedule and the buffer
reconstruction -- is :class:`rl_agents.replay_q_agent.ReplayQAgent`, shared with
``double_q``. What is left here is the vanilla target::

    vanilla DQN:   target = r + discount * max_a' Q_target(s', a')
    double-Q:      target = r + discount * Q_target(s', argmax_a' Q_online(s', a'))

The single ``max`` both selects the bootstrap action and values it, which is the
overestimation bias :mod:`rl_agents.double_q` removes; that one expression is the whole
difference between this agent and the two double-Q variants.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from flax.typing import VariableDict
from rl_components.structs import chex_struct

from rl_agents.q_networks import make_q_network
from rl_agents.replay_q_agent import QApplyFn, ReplayQAgent


@chex_struct(frozen=True, kw_only=True)
class DQNConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    TARGET_NETWORK_FREQUENCY: int = 1_000
    TAU: float = 1.0  # Soft update
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42
    NETWORK_PRESET: Literal["mlp", "nature_cnn"] = "mlp"


def dqn_loss(
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
    """Mean squared error against the vanilla DQN target, over one replay minibatch.

    Reachable from outside the agent so the termination semantics can be gated on the
    real loss rather than on a copy of it. ``discounts`` carries the bootstrap
    coefficient the loop computed, so a terminal row's ``0.0`` removes the bootstrap term
    outright and the loss becomes independent of ``next_obs`` on that row.

    Args:
        params: Online parameters. Differentiate with respect to this argument only.
        target_params: Target-network parameters, which both select and value the
            bootstrap action -- the conflation double-Q splits apart.
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

    next_q_max = jnp.max(apply_fn(target_params, next_obs), axis=-1)

    target = rewards + discounts * next_q_max
    return jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))


class DQNAgent(ReplayQAgent):
    """Deep Q-learning with a target network and epsilon-greedy exploration."""

    def __init__(self, config: DQNConfig) -> None:
        """Bind the configuration, the preset-selected Q-network and the vanilla target.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        super().__init__(
            config,
            lambda action_dim, observation_shape: make_q_network(config, action_dim, observation_shape=observation_shape),
            dqn_loss,
        )
