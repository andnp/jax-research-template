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
from flax.typing import VariableDict
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


def expected_action_value(q_next: jax.Array, epsilon: jax.Array | float) -> jax.Array:
    """Epsilon-greedy expected next-state value, uniform over the argmax set.

    The greedy distribution is stop-gradiented, so the bootstrap contributes
    gradient only through the action values themselves.
    """
    n_actions = q_next.shape[-1]
    greedy = (q_next == q_next.max()).astype(q_next.dtype)
    pi = greedy / greedy.sum()
    pi = (1.0 - epsilon) * pi + epsilon / n_actions
    pi = jax.lax.stop_gradient(pi)
    return q_next.dot(pi)


def qrc_loss(
    q: jax.Array,
    h: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    gamma: jax.Array,
    q_next: jax.Array,
    epsilon: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-transition QRC loss (Ghiassian et al. 2020, eqs. 8-9).

    ``gamma * sg(delta_hat) * v_next`` is the gradient-TD correction term:
    differentiating it w.r.t. the online parameters yields
    ``gamma * delta_hat * grad(v_next)``, which is exactly the term that
    distinguishes gradient TD from semi-gradient TD and removes the bias
    responsible for divergence under off-policy bootstrapping (e.g. DQN).
    """
    v_next = expected_action_value(q_next, epsilon)
    target = jax.lax.stop_gradient(reward + gamma * v_next)

    delta = target - q[action]
    delta_hat = h[action]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * v_next
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return v_loss, h_loss, delta


def qrc_loss_batch(
    params: VariableDict,
    network: QRCNetwork,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    dones: jax.Array,
    gamma: float,
    epsilon: float,
    beta: float,
) -> jax.Array:
    """Mean QRC loss over a minibatch, plus L2 regularisation on the h-head.

    Regularising only the h-head (Ghiassian et al. 2020, §3.2) keeps the
    correction term bounded without biasing the q-head's value estimates.
    """
    q, h = network.apply(params, obs)
    q_next, _ = network.apply(params, next_obs)
    gammas = gamma * (1.0 - dones)

    v_loss, h_loss, _ = jax.vmap(qrc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
        q, h, actions, rewards, gammas, q_next, epsilon
    )

    h_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params["params"]["h_head"]))
    return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * h_reg
