"""Preprocess timesteps between an environment and the agent that consumes them.

Experiments opt in by wrapping an agent, for example to clip rewards. The loop records raw
environment rewards before passing a timestep to the wrapper, so loop metrics stay raw.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import chex
import jax

from rl_components.agent_protocol import AgentProtocol, AgentStep
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

type TimestepTransform[ObsT] = Callable[[Timestep[ObsT]], Timestep[ObsT]]


class PreprocessedAgent[AgentStateT, ObsT, ActT]:
    """Apply timestep transforms before forwarding agent operations."""

    def __init__(
        self,
        agent: AgentProtocol[AgentStateT, ObsT, ActT],
        transforms: Sequence[TimestepTransform[ObsT]],
    ) -> None:
        self._agent = agent
        self._transforms = tuple(transforms)

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> AgentStateT:
        return self._agent.init(key, spec)

    def step(
        self,
        state: AgentStateT,
        timestep: Timestep[ObsT],
        step_index: jax.Array,
    ) -> AgentStep[AgentStateT, ActT]:
        for transform in self._transforms:
            timestep = transform(timestep)
        return self._agent.step(state, timestep, step_index)
