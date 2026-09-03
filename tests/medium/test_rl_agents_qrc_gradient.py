"""Medium tests for rl_agents.qrc — gradient flow, JIT compilation, and the stop-gradient contract."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_agents.qrc import QRCNetwork, qrc_loss, qrc_loss_batch


def _batch(key: jax.Array, obs_dim: int, action_dim: int, batch_size: int) -> tuple[jax.Array, ...]:
    keys = jax.random.split(key, 4)
    obs = jax.random.normal(keys[0], (batch_size, obs_dim))
    actions = jax.random.randint(keys[1], (batch_size,), 0, action_dim)
    rewards = jax.random.normal(keys[2], (batch_size,))
    next_obs = jax.random.normal(keys[3], (batch_size, obs_dim))
    dones = jnp.zeros((batch_size,))
    return obs, actions, rewards, next_obs, dones


class TestQRCGradientFlow:
    def test_params_change_after_update(self) -> None:
        net = QRCNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        tx = optax.adam(3e-4)
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        obs, actions, rewards, next_obs, dones = _batch(jax.random.key(1), 4, 2, 32)

        def loss_fn(params: object) -> jax.Array:
            return qrc_loss_batch(params, net, obs, actions, rewards, next_obs, dones, 0.99, 0.1, 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        new_state = train_state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(train_state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed, "Parameters did not change after gradient step"
        assert jnp.isfinite(loss)

    def test_loss_fn_jit(self) -> None:
        net = QRCNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        obs, actions, rewards, next_obs, dones = _batch(jax.random.key(1), 4, 2, 8)

        @jax.jit
        def compute_loss(params: object) -> jax.Array:
            return qrc_loss_batch(params, net, obs, actions, rewards, next_obs, dones, 0.99, 0.1, 1.0)

        loss = compute_loss(params)
        assert loss.shape == ()

    def test_gradient_correction_term_reaches_trunk_parameters(self) -> None:
        """The gamma*sg(h)*v_next term must change the trunk gradient when h is nonzero.

        The heads are zero-initialised, which would make the q-head's own
        gradient trivially zero regardless of h — so this test replaces the
        heads with random weights first, isolating the effect of h alone.
        """
        net = QRCNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        obs, actions, rewards, next_obs, dones = _batch(jax.random.key(1), 4, 2, 16)

        q_head_kernel = params["params"]["q_head"]["kernel"]
        nonzero_q_head = jax.random.normal(jax.random.key(3), q_head_kernel.shape)
        params = {
            "params": {
                **params["params"],
                "q_head": {**params["params"]["q_head"], "kernel": nonzero_q_head},
            }
        }

        h_head_kernel = params["params"]["h_head"]["kernel"]
        nonzero_h_head = jax.random.normal(jax.random.key(2), h_head_kernel.shape)
        params_h_nonzero = {
            "params": {**params["params"], "h_head": {"kernel": nonzero_h_head}}
        }

        def trunk_loss(params: object) -> jax.Array:
            return qrc_loss_batch(params, net, obs, actions, rewards, next_obs, dones, 0.99, 0.1, 0.0)

        grad_zero = jax.grad(trunk_loss)(params)
        grad_nonzero = jax.grad(trunk_loss)(params_h_nonzero)

        trunk_zero = grad_zero["params"]["Dense_1"]["kernel"]
        trunk_nonzero = grad_nonzero["params"]["Dense_1"]["kernel"]
        assert not jnp.allclose(trunk_zero, trunk_nonzero), "h-head value must change the trunk gradient via the correction term"

    def test_h_head_gradient_does_not_reach_trunk(self) -> None:
        """The h-head reads stop-gradiented features: its loss must not shape the trunk or q-head.

        The h-head is zero-initialised, which would trivially zero this
        gradient via the chain rule regardless of the stop-gradient — so this
        test replaces the h-head kernel with random weights first.
        """
        net = QRCNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        obs, actions, rewards, next_obs, dones = _batch(jax.random.key(1), 4, 2, 16)

        h_head_kernel = params["params"]["h_head"]["kernel"]
        nonzero_h_head = jax.random.normal(jax.random.key(4), h_head_kernel.shape)
        params = {"params": {**params["params"], "h_head": {"kernel": nonzero_h_head}}}

        q, h = net.apply(params, obs)
        q_next, _ = net.apply(params, next_obs)
        gammas = 0.99 * (1.0 - dones)

        def h_loss_only(params: object) -> jax.Array:
            q, h = net.apply(params, obs)
            q_next, _ = net.apply(params, next_obs)
            _, h_loss, _ = jax.vmap(qrc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
                q, h, actions, rewards, gammas, q_next, 0.1
            )
            return jnp.mean(h_loss)

        grads = jax.grad(h_loss_only)(params)

        for layer in ("Dense_0", "Dense_1", "q_head"):
            leaves = jax.tree_util.tree_leaves(grads["params"][layer])
            assert all(jnp.allclose(leaf, 0.0) for leaf in leaves), f"h-loss must not reach {layer}"

        h_head_leaves = jax.tree_util.tree_leaves(grads["params"]["h_head"])
        assert any(jnp.any(leaf != 0.0) for leaf in h_head_leaves), "h-loss must reach the h-head"

    def test_terminal_transitions_zero_bootstrap_in_batch(self) -> None:
        """A done=1 transition must ignore next_obs entirely, regardless of its value."""
        net = QRCNetwork(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        obs, actions, rewards, _, _ = _batch(jax.random.key(1), 4, 2, 8)
        dones = jnp.ones((8,))

        next_obs_a = jax.random.normal(jax.random.key(2), (8, 4))
        next_obs_b = jax.random.normal(jax.random.key(3), (8, 4)) * 100.0

        loss_a = qrc_loss_batch(params, net, obs, actions, rewards, next_obs_a, dones, 0.99, 0.1, 1.0)
        loss_b = qrc_loss_batch(params, net, obs, actions, rewards, next_obs_b, dones, 0.99, 0.1, 1.0)

        assert jnp.allclose(loss_a, loss_b)

    def test_td_loss_is_semi_gradient_apart_from_the_correction_term(self) -> None:
        """With h zeroed, the loss gradient must be exactly ``-delta * grad(q[a])``.

        If the bootstrap target were not stop-gradiented the TD term would become
        a residual gradient, scaling this by ``(1 - gamma)`` — so a self-loop
        transition with ``gamma=0.5`` separates the two by a factor of two.
        """
        action_dim = 1  # v_next == q_next[0], independent of epsilon and tie-breaking
        gamma = 0.5
        net = QRCNetwork(action_dim=action_dim)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))

        # The heads are zero-initialised, which would zero grad(q) through the
        # chain rule and mask any error here; give the q-head a real kernel.
        inner = dict(params["params"])
        inner["q_head"] = {
            "kernel": jax.random.normal(jax.random.key(1), inner["q_head"]["kernel"].shape),
            "bias": inner["q_head"]["bias"],
        }
        params = {**params, "params": inner}

        obs = jax.random.normal(jax.random.key(2), (4,))
        reward = jnp.array(0.7)
        action = jnp.array(0)

        # Self-loop: next_obs == obs, so v_next depends on the same parameters as q.
        batched = (obs[None, :], action[None], reward[None], obs[None, :], jnp.zeros((1,)))

        def total_loss(p: VariableDict) -> jax.Array:
            return qrc_loss_batch(p, net, *batched, gamma, 0.1, 0.0)

        def q_taken(p: VariableDict) -> jax.Array:
            q, _ = net.apply(p, obs)
            return q[action]

        q, h = net.apply(params, obs)
        _, _, delta = qrc_loss(q, h, action, reward, jnp.array(gamma), q, 0.1)

        grad_loss = jax.grad(total_loss)(params)["params"]
        grad_q = jax.grad(q_taken)(params)["params"]

        # The h-head also receives gradient from h_loss, so compare the q-path only.
        for name in ("Dense_0", "Dense_1", "q_head"):
            expected = jax.tree.map(lambda g: -delta * g, grad_q[name])
            actual = grad_loss[name]
            for key in expected:
                assert jnp.allclose(actual[key], expected[key], atol=1e-6), name
