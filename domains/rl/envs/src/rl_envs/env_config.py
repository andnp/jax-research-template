"""Hashable environment configs that build an environment and its params."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from rl_components.env_protocol import EnvProtocol

from rl_envs.gymnax_bridge import make_gymnax_env_by_id
from rl_envs.process_control_bridge import make_process_control_env


@dataclass(frozen=True)
class GymnaxConfig:
    """A registered Gymnax environment, with continuous actions normalized to ``[-1, 1]`` if asked."""

    ENV_ID: str
    NORMALIZED: bool = False

    def build(self) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]:
        return make_gymnax_env_by_id(self.ENV_ID, normalized=self.NORMALIZED)


@dataclass(frozen=True)
class ProcessControlConfig:
    """A process-control benchmark, optionally pinned to one named scenario."""

    ENV_ID: str
    SCENARIO: str | None = None

    def build(self) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], None]:
        overrides = {} if self.SCENARIO is None else {"scenario": self.SCENARIO}
        return make_process_control_env(self.ENV_ID, **overrides), None
