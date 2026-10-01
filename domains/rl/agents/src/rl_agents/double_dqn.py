"""Double DQN (van Hasselt et al., 2016) as an ``AgentProtocol`` implementation.

The update rule lives once in :mod:`rl_agents.double_q`; this module contributes the
public config and the plain Q-network presets. ``dueling_dqn`` contributes the same
config with a dueling network and nothing else, so these two agents differ in their
network alone.
"""

from __future__ import annotations

from typing import Literal

from research_runner.config import traced
from rl_components.structs import chex_struct
from rl_networks.q_networks import make_q_network

from rl_agents.double_q import DoubleQAgent


@chex_struct(frozen=True, kw_only=True)
class DoubleDQNConfig:
    learning_rate: float = traced(3e-4)
    buffer_size: int = 100_000
    batch_size: int = 64
    total_timesteps: int = 200_000
    learning_starts: int = traced(1_000)
    train_frequency: int = traced(1)
    target_network_frequency: int = traced(1_000)
    tau: float = traced(1.0)  # Soft update
    epsilon_start: float = traced(1.0)
    epsilon_end: float = traced(0.05)
    epsilon_fraction: float = traced(0.5)
    env_name: str = "MountainCar-v0"
    seed: int = 42
    network_preset: Literal["mlp", "nature_cnn"] = "mlp"

    def build(self) -> DoubleDQNAgent:
        return DoubleDQNAgent(self)


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
