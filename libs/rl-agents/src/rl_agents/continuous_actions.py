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
reports exactly those bounds. Handing a raw spec straight to one of these agents would
otherwise confine the policy to the middle of the real action range and keep training, so
:func:`continuous_action_dim` refuses a spec whose declared bounds are anything else.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jax.errors import TracerBoolConversionError
from rl_components.env_protocol import EnvSpec


def continuous_action_dim(spec: EnvSpec) -> int:
    """Read the action dimension a tanh-squashed policy needs from ``spec``.

    A spec that declares no bounds is accepted, because there is nothing to disagree with;
    a spec whose bounds are traced is also accepted, since the comparison cannot be made
    at trace time. Both mirror how :class:`~rl_components.env_protocol.EnvSpec` validates
    its own bounds.

    Args:
        spec: The environment description handed to the agent's ``init``.

    Returns:
        The length of the action vector.

    Raises:
        ValueError: If ``spec`` describes a discrete action space, an action of any rank
            other than one, or declared bounds other than ``[-1, 1]``.
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

    if spec.action_low is None or spec.action_high is None:
        return action_shape[0]
    try:
        is_normalized = bool(jnp.all(spec.action_low == -1.0) & jnp.all(spec.action_high == 1.0))
    except TracerBoolConversionError:
        return action_shape[0]
    if not is_normalized:
        raise ValueError(
            f"a tanh-squashed policy requires action bounds of [-1, 1], got spec {spec.id!r} with "
            f"low {spec.action_low} and high {spec.action_high}; wrap the environment in "
            "rl_components.action_normalization.make_action_normalization_wrapper"
        )
    return action_shape[0]


def uniform_exploration_action(key: chex.PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    """Sample the warmup action: uniform over the ``[-1, 1]`` box the policy also lives in."""
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0).astype(dtype)
