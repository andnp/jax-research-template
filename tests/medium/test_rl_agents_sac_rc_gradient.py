"""Medium tests for rl_agents.sac_rc — gradient flow and the stop-gradient contract.

These are the load-bearing tests for SAC-RC's defining property: the
gradient-TD correction term's bootstrap must carry gradient through the
ONLINE critic parameters (no target network). Both tests difference real
gradients of ``sac_rc_loss`` rather than re-deriving the production
expression, so a mutation that breaks the stop-gradient contract is actually
caught.
"""

import jax
import jax.numpy as jnp
from flax.typing import VariableDict
from rl_agents.sac import Actor
from rl_agents.sac_rc import SACRCCritic, sac_rc_loss, sac_rc_loss_batch


def _batch(key: jax.Array, obs_dim: int, action_dim: int, batch_size: int, discount: float = 0.99) -> tuple[jax.Array, ...]:
    keys = jax.random.split(key, 4)
    obs = jax.random.normal(keys[0], (batch_size, obs_dim))
    actions = jax.random.uniform(keys[1], (batch_size, action_dim), minval=-1.0, maxval=1.0)
    rewards = jax.random.normal(keys[2], (batch_size,))
    next_obs = jax.random.normal(keys[3], (batch_size, obs_dim))
    discounts = jnp.full((batch_size,), discount)
    return obs, actions, rewards, next_obs, discounts


class TestSACRCGradientFlow:
    def test_params_change_after_update(self) -> None:
        critic = SACRCCritic()
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
            jax.random.split(jax.random.key(0), 2), jnp.zeros(4), jnp.zeros(2)
        )
        actor = Actor(2)
        actor_params = actor.init(jax.random.key(1), jnp.zeros(4))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 4, 2, 16)

        def loss_fn(params: object) -> jax.Array:
            return sac_rc_loss_batch(
                params, critic, actor_params, actor, jnp.array(0.2),
                obs, actions, rewards, next_obs, discounts, jax.random.key(3), 1.0,
            )

        loss, grads = jax.value_and_grad(loss_fn)(critic_params)
        assert jnp.isfinite(loss)
        flat = jax.tree_util.tree_leaves(grads)
        assert any(jnp.any(g != 0.0) for g in flat)

    def test_loss_fn_jit(self) -> None:
        critic = SACRCCritic()
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
            jax.random.split(jax.random.key(0), 2), jnp.zeros(4), jnp.zeros(2)
        )
        actor = Actor(2)
        actor_params = actor.init(jax.random.key(1), jnp.zeros(4))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(2), 4, 2, 8)

        @jax.jit
        def compute_loss(params: object) -> jax.Array:
            return sac_rc_loss_batch(
                params, critic, actor_params, actor, jnp.array(0.2),
                obs, actions, rewards, next_obs, discounts, jax.random.key(3), 1.0,
            )

        loss = compute_loss(critic_params)
        assert loss.shape == ()

    def test_correction_term_reaches_online_bootstrap(self) -> None:
        """grad(L_Q) must differ between a zeroed and a nonzero h-head.

        The h-head is zero-initialised, so at init the correction term
        ``discount * sg(h) * bootstrap`` is identically zero and contributes
        no gradient regardless of whether the bootstrap carries gradient.
        Giving h nonzero values makes the correction term's presence
        observable: ``bootstrap`` (``next_q_min - alpha*next_log_prob``) is
        computed by applying the SAME critic params to ``next_obs``, so if
        the correction term reaches that online bootstrap, changing h must
        change the trunk/q-head gradient (which also produces the bootstrap).

        MUTATION CHECK: wrapping the correction term's bootstrap in
        ``jax.lax.stop_gradient`` inside ``sac_rc_loss`` makes this test
        fail, since the correction term would then contribute zero gradient
        to the trunk/q-head regardless of h.
        """
        critic = SACRCCritic()
        params = critic.init(jax.random.key(0), jnp.zeros(4), jnp.zeros(2))
        obs, actions, rewards, next_obs, discounts = _batch(jax.random.key(1), 4, 2, 16)
        next_actions = jax.random.uniform(jax.random.key(5), (16, 2), minval=-1.0, maxval=1.0)
        next_log_probs = jnp.zeros(16)
        alpha = jnp.array(0.2)

        h_head_kernel = params["params"]["h_head"]["kernel"]
        nonzero_h_head = jax.random.normal(jax.random.key(4), h_head_kernel.shape)
        params_h_nonzero = {"params": {**params["params"], "h_head": {"kernel": nonzero_h_head}}}

        def batch_loss(p: VariableDict) -> jax.Array:
            q, h = jax.vmap(lambda o, a: critic.apply(p, o, a))(obs, actions)
            next_q, _ = jax.vmap(lambda o, a: critic.apply(p, o, a))(next_obs, next_actions)
            v_loss, _, _ = jax.vmap(sac_rc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
                q, h, rewards, discounts, next_q, next_log_probs, alpha
            )
            return jnp.mean(v_loss)

        grad_zero = jax.grad(batch_loss)(params)["params"]
        grad_nonzero = jax.grad(batch_loss)(params_h_nonzero)["params"]

        trunk_zero = grad_zero["Dense_1"]["kernel"]
        trunk_nonzero = grad_nonzero["Dense_1"]["kernel"]
        assert not jnp.allclose(trunk_zero, trunk_nonzero), (
            "h-head value must change the trunk gradient via the correction term's online bootstrap"
        )

        q_head_zero = grad_zero["q_head"]["kernel"]
        q_head_nonzero = grad_nonzero["q_head"]["kernel"]
        assert not jnp.allclose(q_head_zero, q_head_nonzero), (
            "h-head value must change the q-head gradient via the correction term's online bootstrap"
        )

    def test_td_part_is_semi_gradient_apart_from_correction(self) -> None:
        """With h zeroed, grad(L_Q) must equal ``-delta * grad(Q(s,a))`` exactly.

        Self-loop transition (``next_obs == obs``, bootstrap read from the
        same ``Q(obs, action)`` call) isolates the semi-gradient TD term from
        the (here inert, since h == 0) correction term.

        MUTATION CHECK: removing the ``jax.lax.stop_gradient`` around
        ``target`` (equivalently around ``delta``, since ``h`` is zero so the
        h-loss term is unaffected) in ``sac_rc_loss`` makes this test fail,
        since the TD term would then pick up an extra residual-gradient
        contribution through the target's dependence on the online Q.
        """
        critic = SACRCCritic()
        params = critic.init(jax.random.key(0), jnp.zeros(4), jnp.zeros(2))
        discount = jnp.array(0.5)
        obs = jax.random.normal(jax.random.key(2), (4,))
        action = jax.random.uniform(jax.random.key(3), (2,), minval=-1.0, maxval=1.0)
        reward = jnp.array(0.7)
        alpha = jnp.array(0.2)

        def total_loss(p: VariableDict) -> jax.Array:
            # Self-loop: bootstrap comes from the same Q(obs, action) call,
            # so it shares parameters with the Q(s,a) being regressed.
            q, h = critic.apply(p, obs, action)
            v_loss, h_loss, _ = sac_rc_loss(q, h, reward, discount, q, jnp.array(0.0), alpha)
            return v_loss + h_loss

        def q_taken(p: VariableDict) -> jax.Array:
            q, _ = critic.apply(p, obs, action)
            return q

        q, h = critic.apply(params, obs, action)
        _, _, delta = sac_rc_loss(q, h, reward, discount, q, jnp.array(0.0), alpha)

        grad_loss = jax.grad(total_loss)(params)["params"]
        grad_q = jax.grad(q_taken)(params)["params"]

        for name in ("Dense_0", "Dense_1", "q_head"):
            for key in grad_q[name]:
                expected = -delta * grad_q[name][key]
                actual = grad_loss[name][key]
                assert jnp.allclose(actual, expected, atol=1e-6), name
