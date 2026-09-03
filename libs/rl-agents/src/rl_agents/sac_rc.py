"""Soft Actor-Critic with Regularized Corrections (SAC-RC) for continuous-action control.

The continuous-action analogue of ``rl_agents.qrc``: SAC's twin-critic
ensemble gets the same TDRC/QRC gradient-TD correction QRC applies to DQN
(Ghiassian et al., "Gradient Temporal-Difference Learning with Regularized
Corrections", ICML 2020, arXiv:2007.00611).

Two design decisions carried over from ``qrc.py``:

- The h-head reads stop-gradiented trunk features — it is a linear probe on
  frozen features and never shapes the trunk/q-head representation.
- There is no target network. SAC-RC is pure gradient TD: every bootstrap
  uses the online critic parameters, and the ``gamma * sg(h) * bootstrap``
  term in the loss supplies the gradient correction a target network would
  otherwise stand in for. Stop-gradienting that bootstrap would silently
  collapse SAC-RC back into plain semi-gradient SAC with an inert extra head.

The actor loss and the entropy-coefficient (alpha) update are unchanged from
``rl_agents.sac``; only the critic loss and the twin-critic network gain the
h-head and correction term.
"""

from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import VariableDict
from rl_components.structs import chex_struct

from rl_agents.sac import Actor


@chex_struct(frozen=True, kw_only=True)
class SACRCConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 5_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    ALPHA: float = 0.2
    TARGET_ENTROPY: float | None = None
    BETA: float = 1.0
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


class SACRCCritic(nn.Module):
    """Twin-critic trunk (matching ``sac.Critic``) plus a QRC-style h-head.

    The h-head reads ``jax.lax.stop_gradient``-ed trunk features, and is
    zero-initialised and bias-free, exactly as in ``qrc.QRCNetwork`` — it is
    a linear probe on frozen features that starts at zero and never shapes
    the trunk. The trunk and q-head match ``sac.Critic`` (default init).
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        xa = jnp.concatenate([x, a], axis=-1)
        phi = nn.Dense(256)(xa)
        phi = nn.relu(phi)
        phi = nn.Dense(256)(phi)
        phi = nn.relu(phi)
        q = nn.Dense(1, name="q_head")(phi)
        h = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            use_bias=False,
            name="h_head",
        )(jax.lax.stop_gradient(phi))
        return jnp.squeeze(q, axis=-1), jnp.squeeze(h, axis=-1)


def _critic_apply(module: SACRCCritic, variables: VariableDict, x: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return cast(tuple[jax.Array, jax.Array], module.apply(variables, x, a))


def sac_rc_loss(
    q: jax.Array,
    h: jax.Array,
    reward: jax.Array,
    discount: jax.Array,
    next_q_min: jax.Array,
    next_log_prob: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-transition, per-ensemble-member SAC-RC loss.

    ``bootstrap`` is the soft value ``min_j Q_j(s', a') - alpha * log pi(a'|s')``
    computed from the ONLINE critic (no target network). ``target`` stop-
    gradients that bootstrap for the semi-gradient TD term, but the
    correction term ``gamma * sg(h) * bootstrap`` reuses the un-stopped
    ``bootstrap`` so its gradient reaches the online parameters that produced
    it: differentiating it w.r.t. those parameters gives
    ``gamma * delta_hat * grad(bootstrap)``, the term that distinguishes
    gradient TD from semi-gradient TD and removes the divergence-inducing
    off-policy bootstrapping bias that a target network is normally used to
    paper over.
    """
    target = jax.lax.stop_gradient(reward + discount * (next_q_min - alpha * next_log_prob))
    delta = target - q
    delta_hat = h

    v_loss = 0.5 * delta**2 + discount * jax.lax.stop_gradient(delta_hat) * (next_q_min - alpha * next_log_prob)
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return v_loss, h_loss, delta


def sac_rc_loss_batch(
    critic_params: VariableDict,
    critic: SACRCCritic,
    actor_params: VariableDict,
    actor: Actor,
    alpha: jax.Array,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    rng: jax.Array,
    beta: float,
) -> jax.Array:
    """Mean SAC-RC critic loss over a minibatch, plus L2 regularisation on the h-heads.

    ``next_q_min`` is shared across ensemble members (the min is over the
    ensemble), each member's own ``q``/``h`` are read from its own params via
    ``vmap``, matching the twin-critic structure of ``sac.py``.
    """
    next_actions, next_log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, next_obs, jax.random.split(rng, obs.shape[0])
    )

    next_q_values = jax.vmap(
        lambda params, o, a: _critic_apply(critic, params, o, a)[0], in_axes=(0, None, None)
    )(critic_params, next_obs, next_actions)
    next_q_min = jnp.min(next_q_values, axis=0)

    def _single_critic_loss(params: VariableDict) -> jax.Array:
        q, h = _critic_apply(critic, params, obs, actions)
        v_loss, h_loss, _ = jax.vmap(sac_rc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
            q, h, rewards, discounts, next_q_min, next_log_probs, alpha
        )
        h_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params["params"]["h_head"]))
        return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * h_reg

    return jnp.mean(jax.vmap(_single_critic_loss)(critic_params))
