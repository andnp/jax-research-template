"""Double DQN (van Hasselt et al., 2016) as an ``AgentProtocol`` implementation.

The update rule lives once in :mod:`rl_agents.double_q`; this module contributes the
public config and the plain Q-network presets. ``dueling_dqn`` contributes the same
config with a dueling network and nothing else, so these two agents differ in their
network alone.
"""

from __future__ import annotations

from typing import Literal

from rl_components.structs import chex_struct

from rl_agents.double_q import DoubleQAgent
from rl_agents.q_networks import make_q_network


@chex_struct(frozen=True, kw_only=True)
class DoubleDQNConfig:
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


class DoubleDQNAgent(DoubleQAgent):
    """The double-Q update over the Q-network presets shared with vanilla DQN."""

    def __init__(self, config: DoubleDQNConfig) -> None:
        """Bind the configuration and the preset-selected Q-network.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar.
        """
        super().__init__(
            config,
            lambda action_dim, observation_shape: make_q_network(config, action_dim, observation_shape=observation_shape),
        )
