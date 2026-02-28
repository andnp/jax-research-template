"""Output heads for RL networks (dueling, epsilon-greedy)."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from jax_nn.initializers import output_orthogonal, stable_orthogonal


class DuelingHead(nn.Module):
    """Dueling network head (Wang et al., 2016).

    Separates state-value and advantage streams:

    .. math::

        Q(s, a) = V(s) + A(s, a) - \\frac{1}{|\\mathcal{A}|}
                  \\sum_{a'} A(s, a')

    Each stream has one hidden layer followed by a linear projection.

    Attributes:
        action_dim: Number of discrete actions.
        hidden_features: Width of each stream's hidden layer.
        dtype: Computation dtype.
    """

    action_dim: int
    hidden_features: int = 512
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Compute Q-values from shared features.

        Args:
            x: Feature tensor, shape ``(..., feature_dim)``.

        Returns:
            Q-values, shape ``(..., action_dim)``.
        """
        v = nn.Dense(
            self.hidden_features,
            kernel_init=stable_orthogonal(),
            dtype=self.dtype,
        )(x)
        v = nn.relu(v)
        v = nn.Dense(1, kernel_init=output_orthogonal(), dtype=self.dtype)(v)

        a = nn.Dense(
            self.hidden_features,
            kernel_init=stable_orthogonal(),
            dtype=self.dtype,
        )(x)
        a = nn.relu(a)
        a = nn.Dense(
            self.action_dim,
            kernel_init=output_orthogonal(),
            dtype=self.dtype,
        )(a)

        return v + (a - jnp.mean(a, axis=-1, keepdims=True))


def epsilon_greedy_action(
    q_values: Array,
    epsilon: Array | float,
    *,
    key: Array,
) -> Array:
    """Select an action using epsilon-greedy policy (JIT-compatible).

    With probability ``epsilon``, selects a uniform random action.
    Otherwise, selects the greedy (argmax Q) action.

    Uses ``jnp.where`` for JIT compatibility — both branches are
    evaluated but only one result is returned.

    Args:
        q_values: Q-value predictions, shape ``(num_actions,)``
            or ``(batch, num_actions)``.
        epsilon: Exploration rate in ``[0, 1]``. Can be a scalar or
            a JAX traced value (e.g., from a schedule).
        key: PRNG key for random action sampling.
    """
    key_choice, key_action = jax.random.split(key)
    num_actions = q_values.shape[-1]
    greedy_action = jnp.argmax(q_values, axis=-1)
    random_action = jax.random.randint(key_action, greedy_action.shape, 0, num_actions)
    is_greedy = jax.random.uniform(key_choice, greedy_action.shape) >= epsilon
    return jnp.where(is_greedy, greedy_action, random_action)
