"""Q-learning with Regularized Corrections (QRC) for discrete-action control.

Ghiassian et al., "Gradient Temporal-Difference Learning with Regularized
Corrections", ICML 2020 (arXiv:2007.00611). QRC is the nonlinear control
variant of TDRC: an auxiliary h-head corrects the semi-gradient bias that can
make DQN-style bootstrapping diverge.

Two design decisions carried over from the reference implementation:

- The h-head reads stop-gradiented trunk features (``addHead(..., grad=False)``
  in the reference code) — it is a linear probe on frozen features and never
  shapes the trunk representation.
- There is no target network. QRC is pure gradient TD: the bootstrap uses the
  online parameters, and the ``gamma * sg(h) * v_next`` term in the loss
  supplies the gradient correction a target network would otherwise stand in
  for.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_nn.typed_module import TypedApply
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class QRCConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    BETA: float = 1.0
    ENV_NAME: str = "CartPole-v1"
    SEED: int = 42


class QRCNetwork(TypedApply[tuple[jax.Array, jax.Array]], nn.Module):
    """Shared trunk with zero-initialised linear q- and h-heads.

    The h-head reads ``jax.lax.stop_gradient``-ed trunk features, so only the
    q-head's loss shapes the trunk; the h-head is a linear probe on frozen
    features.
    """

    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        phi = nn.Dense(64)(x)
        phi = nn.relu(phi)
        phi = nn.Dense(64)(phi)
        phi = nn.relu(phi)
        q = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="q_head",
        )(phi)
        h = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            use_bias=False,
            name="h_head",
        )(jax.lax.stop_gradient(phi))
        return q, h
