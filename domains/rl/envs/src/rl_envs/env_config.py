"""Hashable environment configs that build an environment and its params."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from rl_components.env_protocol import EnvProtocol

from rl_envs.gymnax_bridge import make_gymnax_env_by_id


@dataclass(frozen=True)
class GymnaxConfig:
    """A registered Gymnax environment, with continuous actions normalized to ``[-1, 1]`` if asked."""

    ENV_ID: str
    NORMALIZED: bool = False

    def build(self) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]:
        return make_gymnax_env_by_id(self.ENV_ID, normalized=self.NORMALIZED)
