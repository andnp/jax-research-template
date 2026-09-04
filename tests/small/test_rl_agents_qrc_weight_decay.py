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
from rl_agents.qrc import QRCNetwork, _make_tx


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
