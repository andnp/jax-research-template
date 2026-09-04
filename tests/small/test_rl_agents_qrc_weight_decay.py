"""Deviation 1: the h-head gets decoupled AdamW weight decay, the trunk/q-head do not.

A zero gradient with nonzero weight decay must still shrink AdamW-managed
parameters (decoupled decay acts even without a loss gradient), while plain
Adam-managed parameters are exactly unchanged. This exercises the real
``_make_tx``/``optax.multi_transform`` construction rather than re-deriving
it, and is the required mutation-checked test for Deviation 1: giving both
groups plain Adam (verified by hand against the source) makes the h-head
leaves stop shrinking, failing the ``h_shrunk`` assertion below.
"""

from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax.typing import VariableDict
from rl_agents.qrc import QRCNetwork, _make_tx, qrc_loss_batch


def test_h_head_shrinks_under_zero_gradient_but_trunk_and_q_head_do_not() -> None:
    net = QRCNetwork(action_dim=3)
    raw_params = net.init(jax.random.key(0), jnp.zeros((4,)))
    # Zero-init heads never move on their own; give every leaf a real value.
    params = cast(VariableDict, jax.tree.map(lambda leaf: leaf + 1.0, raw_params))

    weight_decay = 0.5
    tx = _make_tx(params, learning_rate=1e-2, weight_decay=weight_decay)
    opt_state = tx.init(params)

    zero_grads = jax.tree.map(jnp.zeros_like, params)
    updates, _ = tx.update(zero_grads, opt_state, params)
    new_params = cast(VariableDict, optax.apply_updates(params, updates))

    h_before = params["params"]["h_head"]["kernel"]
    h_after = new_params["params"]["h_head"]["kernel"]
    assert not jnp.allclose(h_before, h_after), "h-head must shrink under decoupled weight decay even with zero gradient"

    for name in ("Dense_0", "Dense_1", "q_head"):
        for key in params["params"][name]:
            before = params["params"][name][key]
            after = new_params["params"][name][key]
            assert jnp.allclose(before, after, atol=1e-8), f"{name}.{key} must not move under a zero gradient (plain Adam)"


def test_h_head_does_not_shrink_under_zero_gradient_in_l2_mode() -> None:
    """``h_regularization="l2"`` must give the h-head plain Adam, not AdamW.

    Mutation check: if ``_make_tx`` still routed "l2" mode through
    ``optax.adamw``, the h-head would shrink under a zero gradient exactly as
    in the AdamW-mode test above, failing this assertion.
    """
    net = QRCNetwork(action_dim=3)
    raw_params = net.init(jax.random.key(0), jnp.zeros((4,)))
    params = cast(VariableDict, jax.tree.map(lambda leaf: leaf + 1.0, raw_params))

    tx = _make_tx(params, learning_rate=1e-2, weight_decay=0.5, h_regularization="l2")
    opt_state = tx.init(params)

    zero_grads = jax.tree.map(jnp.zeros_like, params)
    updates, _ = tx.update(zero_grads, opt_state, params)
    new_params = cast(VariableDict, optax.apply_updates(params, updates))

    h_before = params["params"]["h_head"]["kernel"]
    h_after = new_params["params"]["h_head"]["kernel"]
    assert jnp.allclose(h_before, h_after, atol=1e-8), "h-head must not move under a zero gradient in l2 mode (plain Adam)"


def test_l2_mode_adds_beta_scaled_h_head_penalty_to_the_loss() -> None:
    """``h_regularization="l2"`` must add ``beta * sum(h_head**2)`` to the loss.

    Non-zero, non-symmetric inputs throughout (observations, actions, rewards)
    so the base QRC loss itself is nonzero and this isn't a vacuous
    zero-plus-zero comparison. Mutation check: dropping the ``if
    h_regularization == "l2"`` branch (always returning the base loss) makes
    the two losses below equal, failing the strict inequality.
    """
    net = QRCNetwork(action_dim=3)
    key = jax.random.key(0)
    params = cast(VariableDict, net.init(key, jnp.zeros((4,))))
    # Zero-init heads never move on their own; give the h-head a real, nonzero value.
    params = cast(
        VariableDict,
        {**params, "params": {**params["params"], "h_head": jax.tree.map(lambda leaf: leaf + 0.3, params["params"]["h_head"])}},
    )

    obs_key, next_key = jax.random.split(key)
    obs = jax.random.normal(obs_key, (5, 4))
    next_obs = jax.random.normal(next_key, (5, 4))
    actions = jnp.array([0, 1, 2, 0, 1])
    rewards = jnp.array([1.0, -1.0, 0.5, 2.0, -0.5])
    discounts = jnp.full((5,), 0.99)

    kwargs = dict(
        params=params, obs=obs, actions=actions, rewards=rewards, next_obs=next_obs,
        discounts=discounts, epsilon=0.1, apply_fn=net.apply,
    )
    base_loss = qrc_loss_batch(**kwargs, h_regularization="adamw", beta=2.0)
    l2_loss_small_beta = qrc_loss_batch(**kwargs, h_regularization="l2", beta=2.0)
    l2_loss_large_beta = qrc_loss_batch(**kwargs, h_regularization="l2", beta=5.0)

    # adamw mode ignores beta entirely (regularisation lives in the optimizer), so the
    # l2 loss must exceed it, and must keep growing as beta grows.
    assert l2_loss_small_beta > base_loss
    assert l2_loss_large_beta > l2_loss_small_beta
