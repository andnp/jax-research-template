"""What the continuous-action agents need from an :class:`~rl_components.env_protocol.EnvSpec`.

Every ported continuous agent squashes its policy through ``tanh``, so it emits a
one-dimensional action vector in ``[-1, 1]`` and nothing else. Two consequences are
shared by all of them and live here rather than in four copies:

- the action dimension, which the agent's ``init`` reads from the spec and its ``step``
  re-reads from the shape of its own carried action, since ``step`` never sees a spec;
- warmup exploration, which is uniform over the same ``[-1, 1]`` box.

The ``[-1, 1]`` range is a requirement on the environment, not a rescaling these agents
perform. An environment with native bounds must be wrapped in
:func:`rl_components.action_normalization.make_action_normalization_wrapper`, whose spec
reports exactly those bounds; handing a raw spec straight to one of these agents silently
confines the policy to the middle of the real action range.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from rl_components.env_protocol import EnvSpec


def continuous_action_dim(spec: EnvSpec) -> int:
    """Read the action dimension a tanh-squashed policy needs from ``spec``.

    Args:
        spec: The environment description handed to the agent's ``init``.

    Returns:
        The length of the action vector.

    Raises:
        ValueError: If ``spec`` describes a discrete action space, or an action of any
            rank other than one.
    """
    if spec.num_actions is not None:
        raise ValueError(
            f"continuous control requires a continuous action space, got spec {spec.id!r} with {spec.num_actions} discrete actions"
        )

    action_shape = tuple(spec.action_shape)
    if len(action_shape) != 1:
        raise ValueError(
            f"continuous control requires a one-dimensional action vector, got spec {spec.id!r} with action_shape {action_shape}"
        )
    return action_shape[0]


def uniform_exploration_action(key: chex.PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    """Sample the warmup action: uniform over the ``[-1, 1]`` box the policy also lives in."""
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0).astype(dtype)
