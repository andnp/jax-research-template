"""Q-network builders shared by the DQN-family agents.

Holds the network classes and observation-layout helpers used by ``dqn``,
``double_dqn``, ``dqn_atari`` and ``rainbow``.
"""

from __future__ import annotations

from typing import Literal, Protocol

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_nn.typed_module import TypedApply

from rl_networks.initializers import legacy_dqn_uniform
from rl_networks.layers import NatureCNN


class QNetwork(TypedApply[jax.Array], nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class HasNetworkPreset(Protocol):
    @property
    def network_preset(self) -> Literal["mlp", "nature_cnn"]: ...


class NatureQNetwork(TypedApply[jax.Array], nn.Module):
    action_dim: int
    observation_layout: Literal["hwc", "fhwc"]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = prepare_nature_observations(x, self.observation_layout)
        x = NatureCNN()(x)
        input_units = x.shape[-1]
        x = nn.Dense(
            512,
            kernel_init=legacy_dqn_uniform(),
            bias_init=legacy_dqn_uniform(num_input_units=input_units),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=legacy_dqn_uniform(),
            bias_init=legacy_dqn_uniform(num_input_units=512),
        )(x)
        return x


def prepare_nature_observations(
    x: jax.Array,
    observation_layout: Literal["hwc", "fhwc"],
) -> jax.Array:
    if observation_layout == "hwc":
        if x.ndim not in (3, 4):
            raise ValueError(
                "network_preset='nature_cnn' with HWC observations expects shape (height, width, channels) or (batch, height, width, channels)."
            )
        return x

    if x.ndim == 4:
        moved = jnp.moveaxis(x, 0, -2)
        return moved.reshape(moved.shape[:-2] + (moved.shape[-2] * moved.shape[-1],))

    if x.ndim == 5:
        moved = jnp.moveaxis(x, 1, -2)
        return moved.reshape(moved.shape[:-2] + (moved.shape[-2] * moved.shape[-1],))

    raise ValueError(
        "network_preset='nature_cnn' with Atari-style observations expects shape (frames, height, width, channels) or (batch, frames, height, width, channels)."
    )


def infer_nature_observation_layout(observation_shape: tuple[int, ...]) -> Literal["hwc", "fhwc"]:
    if len(observation_shape) == 3:
        return "hwc"
    if len(observation_shape) == 4:
        return "fhwc"
    raise ValueError(
        "network_preset='nature_cnn' requires image observations shaped (height, width, channels) or Atari-style (frames, height, width, channels)."
    )


def make_q_network(
    config: HasNetworkPreset,
    action_dim: int,
    observation_shape: tuple[int, ...] | None = None,
) -> QNetwork | NatureQNetwork:
    if config.network_preset == "mlp":
        return QNetwork(action_dim)
    if config.network_preset == "nature_cnn":
        if observation_shape is None:
            raise ValueError("network_preset='nature_cnn' requires observation_shape to build the Q-network.")
        return NatureQNetwork(
            action_dim=action_dim,
            observation_layout=infer_nature_observation_layout(observation_shape),
        )
    raise ValueError(
        f"Invalid network_preset {config.network_preset!r}. Expected one of: 'mlp', 'nature_cnn'."
    )
