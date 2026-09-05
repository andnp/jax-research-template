"""Paper-tied network torsos for RL agents."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jax import Array
from jax_nn.typed_module import TypedApply

from rl_networks.initializers import legacy_dqn_uniform


class NatureCNN(TypedApply[Array], nn.Module):
    """Canonical Atari convolutional torso from the DQN Nature paper.

    Expects channel-last image observations and applies the standard three-layer
    convolutional stack used by DQN Zoo for Atari:

    - ``32`` filters with kernel ``8x8`` and stride ``4``
    - ``64`` filters with kernel ``4x4`` and stride ``2``
    - ``64`` filters with kernel ``3x3`` and stride ``1``

    Inputs are cast to ``float32`` and scaled by ``1 / 255`` before the first
    convolution. The torso ends after the final ReLU and flatten operation; it
    deliberately does not include the DQN head's 512-unit dense layer.

    Attributes:
        dtype: Computation dtype for convolution outputs.
    """

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if x.ndim < 3:
            raise ValueError("NatureCNN expects inputs with shape (..., height, width, channels).")

        x = jnp.asarray(x, dtype=jnp.float32) / jnp.float32(255.0)

        in_channels = x.shape[-1]
        for features, kernel_size, strides in (
            (32, (8, 8), (4, 4)),
            (64, (4, 4), (2, 2)),
            (64, (3, 3), (1, 1)),
        ):
            num_input_units = kernel_size[0] * kernel_size[1] * in_channels
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=strides,
                padding="VALID",
                kernel_init=legacy_dqn_uniform(),
                bias_init=legacy_dqn_uniform(num_input_units=num_input_units),
                dtype=self.dtype,
            )(x)
            x = nn.relu(x)
            in_channels = features

        return x.reshape(x.shape[:-3] + (-1,))
