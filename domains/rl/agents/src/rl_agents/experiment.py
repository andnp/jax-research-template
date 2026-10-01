"""The default RL experiment: one typed config and one JAX train function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
from research_runner.config import traced
from rl_components.agent_protocol import AgentProtocol
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import LoopState, run


class AgentConfig[AgentStateT](Protocol):
    """An agent config: how long to train, and how to build the agent."""

    @property
    def total_timesteps(self) -> int: ...

    def build(self) -> AgentProtocol[AgentStateT, jax.Array, jax.Array]: ...


class EnvConfig(Protocol):
    """An environment config that builds the environment and its params."""

    def build(self) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]: ...


@dataclass(frozen=True)
class RLConfig[AgentStateT]:
    agent: AgentConfig[AgentStateT]
    env: EnvConfig
    gamma: float = traced(0.99)


def rl_train[AgentStateT](
    config: RLConfig[AgentStateT], key: jax.Array
) -> tuple[LoopState[object, AgentStateT, jax.Array], dict[str, jax.Array]]:
    """Build the environment and agent, then run the interaction loop. Metrics are the loop's own."""
    env, env_params = config.env.build()
    return run(
        config.agent.build(),
        env,
        key,
        steps=config.agent.total_timesteps,
        gamma=config.gamma,
        env_params=env_params,
    )
