"""Dueling DQN (Wang et al., 2016) as an ``AgentProtocol`` implementation.

This agent's update is the double-Q target of :mod:`rl_agents.double_q`, unchanged. Its
predecessor shared its whole ``_update_step`` with ``double_dqn`` line for line, so it is
not a dueling variant of vanilla DQN and never was; the only contribution here is the
network, whose separate value and advantage streams recombine as::

    Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))

That decomposition lets the agent learn which states are valuable without learning the
effect of every action in every state.
"""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_nn.heads import DuelingHead
from jax_nn.initializers import stable_orthogonal
from jax_nn.typed_module import TypedApply
from rl_components.structs import chex_struct

from rl_agents.double_q import DoubleQAgent


@chex_struct(frozen=True, kw_only=True)
class DuelingDQNConfig:
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


class DuelingQNetwork(TypedApply[jax.Array], nn.Module):
    """Q-network with a dueling architecture.

    A shared feature extractor feeds into a DuelingHead that
    separates value and advantage streams.
    """

    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64, kernel_init=stable_orthogonal())(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=stable_orthogonal())(x)
        x = nn.relu(x)
        x = DuelingHead(action_dim=self.action_dim, hidden_features=64)(x)
        return x


def _make_dueling_q_network(config: DuelingDQNConfig, action_dim: int) -> DuelingQNetwork:
    """Select the dueling network for ``config``'s preset.

    Takes no observation shape, unlike :func:`rl_agents.q_networks.make_q_network`, which
    needs one only to infer the Nature CNN's frame layout. The one preset supported here
    is a flat MLP, and the agent's ``init`` derives the shape it needs from the
    environment spec.
    """
    if config.NETWORK_PRESET == "mlp":
        return DuelingQNetwork(action_dim)
    if config.NETWORK_PRESET == "nature_cnn":
        raise ValueError(
            "NETWORK_PRESET='nature_cnn' is not yet supported in rl_agents.dueling_dqn because the dueling Nature architecture has not been specified."
        )
    raise ValueError(
        f"Invalid NETWORK_PRESET {config.NETWORK_PRESET!r}. Expected one of: 'mlp', 'nature_cnn'."
    )


class DuelingDQNAgent(DoubleQAgent):
    """The double-Q update over a dueling Q-network."""

    def __init__(self, config: DuelingDQNConfig) -> None:
        """Bind the configuration and the dueling Q-network.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        super().__init__(config, lambda action_dim, _observation_shape: _make_dueling_q_network(config, action_dim))
