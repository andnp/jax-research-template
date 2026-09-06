"""Small tests for rl_agents.gtd_iteration — the iterated-GTD critic correction.

The two parity tests below are load-bearing: they check that :func:`gtd_iteration_correction`
reproduces, gradient-for-gradient (the only thing that determines training behaviour), the
correction formulas iterated-gtd's ``gi_sac.py`` (target-free chain) and ``discrete_gi_qrc.py``
(anchored chain) computed before this module existed -- with one exception, pinned separately
by ``TestFirstBootstrapReference`` below: the target-free reference is the *corrected* formula,
not gi_sac.py's original one, since gi_sac.py's original formula double-counted head 0's
correction (see ``_first_bootstrap_reference``'s docstring) the way discrete_gi_qrc.py's
target-free branch already found and fixed. The anchored case's formula never referenced
``reward`` in its correction term to begin with, so its loss *value* matches exactly too; the
target-free case's semi-gradient form differs by an additive ``reward``/``alpha*log_prob``
constant with zero derivative wrt critic params, so only its gradient is asserted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_agents.gtd_iteration import GTDIterationMetrics, _first_bootstrap_reference, bootstrap_index, gtd_iteration_correction

NUM_HEADS = 4


class TestBootstrapIndex:
    def test_head_zero_self_bootstraps(self) -> None:
        assert bootstrap_index(NUM_HEADS)[0] == 0

    def test_later_heads_bootstrap_off_their_predecessor(self) -> None:
        idx = bootstrap_index(NUM_HEADS)
        assert jnp.array_equal(idx[1:], jnp.arange(NUM_HEADS - 1))


class TestFirstBootstrapReference:
    def test_target_free_chain_masks_only_the_duplicate(self) -> None:
        """Head 1 shares head 0's bootstrap source (both reference index 0); only head 0 corrects it."""
        mask = _first_bootstrap_reference(bootstrap_index(NUM_HEADS))
        assert jnp.array_equal(mask, jnp.array([True, False, True, True]))

    def test_anchored_chain_has_no_duplicates_to_mask(self) -> None:
        """An anchored chain's identity boot_idx has no repeated source, so nothing is masked."""
        mask = _first_bootstrap_reference(jnp.arange(NUM_HEADS, dtype=jnp.int32))
        assert jnp.all(mask)


class TestGtdIterationCorrectionShape:
    def test_metrics_fields(self) -> None:
        n = NUM_HEADS
        own_next_value = jnp.zeros((n,))
        q_online = jnp.zeros((n,))
        h_online = jnp.zeros((n,))
        active = jnp.ones((n,), dtype=bool)
        boot_idx = bootstrap_index(n)
        _, metrics = gtd_iteration_correction(
            own_next_value, q_online, h_online, active, boot_idx, jnp.float32(0.0), jnp.float32(0.9)
        )
        assert isinstance(metrics, GTDIterationMetrics)
        assert metrics.q.shape == (n,)
        assert metrics.delta.shape == (n,)
        assert metrics.bellman_error_sum.shape == ()


def _target_free_reference(
    q_theta: jax.Array,
    h: jax.Array,
    *,
    reward: jax.Array,
    gamma: jax.Array,
    alpha: jax.Array,
    next_log_prob: jax.Array,
    boot_idx: jax.Array,
) -> jax.Array:
    """gi_sac.py's ``_gi_sac_critic_loss``, amended with the duplicate-bootstrap-reference fix.

    ``gi_sac.py``'s original formula gave head 0 both its own self-correction (``H_0``) and
    a downstream correction via ``H_1`` (since head 1 also bootstraps off ``raw_y[0]``),
    double-counting the same source -- the bug ``discrete_gi_qrc.py``'s target-free branch
    had already found and fixed by dropping its analogous ``H_1`` term. This reference
    reflects the corrected formula, matching what :func:`gtd_iteration_correction` now
    computes for both callers. ``q_theta`` stands in for the shared critic parameters: both
    ``q_sa`` and ``q_next`` read the same leaf, matching the real network's single set of
    trainable per-head kernels evaluated at two inputs.
    """
    sg = jax.lax.stop_gradient
    q_sa = q_theta
    q_next = q_theta

    raw_y = reward + gamma * (q_next - alpha * next_log_prob)
    y_target = raw_y[boot_idx]
    delta = sg(y_target) - q_sa

    semi_gradient = -jnp.sum(q_sa * sg(delta))
    self_correction = raw_y[0] * sg(h[0])
    # raw_y[0] already corrected above; head 1 (boot_idx[1] == 0 too) does not correct it again.
    downstream_correction = jnp.sum(raw_y[1:-1] * sg(h[2:]))
    q_loss = semi_gradient + self_correction + downstream_correction
    h_loss = jnp.sum(0.5 * (sg(delta) - h) ** 2)
    return q_loss + h_loss


def _target_free_new(
    q_theta: jax.Array,
    h: jax.Array,
    *,
    reward: jax.Array,
    gamma: jax.Array,
    alpha: jax.Array,
    next_log_prob: jax.Array,
    boot_idx: jax.Array,
) -> jax.Array:
    own_next_value = q_theta - alpha * next_log_prob
    active = jnp.ones_like(h, dtype=bool)
    loss, _ = gtd_iteration_correction(own_next_value, q_theta, h, active, boot_idx, reward, gamma)
    return loss


class TestMatchesGiSacTargetFreeChain:
    def test_gradients_match_the_corrected_target_free_formula(self) -> None:
        q_key, h_key, log_prob_key = jax.random.split(jax.random.key(0), 3)
        q_theta = jax.random.normal(q_key, (NUM_HEADS,))
        h = jax.random.normal(h_key, (NUM_HEADS,))
        next_log_prob = jax.random.normal(log_prob_key, (NUM_HEADS,))
        kwargs = dict(
            reward=jnp.float32(0.3),
            gamma=jnp.float32(0.95),
            alpha=jnp.float32(0.2),
            next_log_prob=next_log_prob,
            boot_idx=bootstrap_index(NUM_HEADS),
        )

        ref_grad_q, ref_grad_h = jax.grad(_target_free_reference, argnums=(0, 1))(q_theta, h, **kwargs)
        new_grad_q, new_grad_h = jax.grad(_target_free_new, argnums=(0, 1))(q_theta, h, **kwargs)

        assert jnp.allclose(ref_grad_q, new_grad_q, atol=1e-5)
        assert jnp.allclose(ref_grad_h, new_grad_h, atol=1e-5)


def _anchored_reference(theta: jax.Array, h: jax.Array, *, target0: jax.Array, reward: jax.Array, gamma: jax.Array) -> jax.Array:
    """discrete_gi_qrc.py's ``_gi_qrc_loss`` anchored branch, before this module existed.

    ``theta`` stands in for the ``K`` online heads' shared value (own current value and
    own next-value read the same trainable leaf); ``target0`` is the frozen target's
    value, a separate array with no trainable-parameter dependence, matching "no
    gradient to correct" for head 0's bootstrap source.
    """
    sg = jax.lax.stop_gradient
    q_online = theta
    v_next_all = jnp.concatenate([target0[None], theta])
    v_next_for_head = v_next_all[:-1]

    y = reward + gamma * v_next_for_head
    delta = sg(y) - q_online
    q_loss = jnp.sum(0.5 * delta**2)

    h_train_target = delta[1:]
    grad_h, grad_v_next = h, v_next_for_head[1:]
    grad_term = jnp.sum(gamma * sg(grad_h) * grad_v_next)
    h_loss = jnp.sum(0.5 * (sg(h_train_target) - h) ** 2)
    return q_loss + grad_term + h_loss


def _anchored_new(theta: jax.Array, h: jax.Array, *, target0: jax.Array, reward: jax.Array, gamma: jax.Array) -> jax.Array:
    own_next_value = jnp.concatenate([target0[None], theta])
    h_online = jnp.concatenate([jnp.zeros((1,)), h])
    active = jnp.concatenate([jnp.zeros((1,), dtype=bool), jnp.ones_like(h, dtype=bool)])
    boot_idx = jnp.arange(theta.shape[0], dtype=jnp.int32)
    loss, _ = gtd_iteration_correction(own_next_value, theta, h_online, active, boot_idx, reward, gamma)
    return loss


class TestMatchesDiscreteGiQrcAnchoredChain:
    def test_values_and_gradients_match_the_original_formula(self) -> None:
        theta_key, h_key, target_key = jax.random.split(jax.random.key(1), 3)
        theta = jax.random.normal(theta_key, (NUM_HEADS,))
        h = jax.random.normal(h_key, (NUM_HEADS - 1,))
        target0 = jax.random.normal(target_key, ())
        kwargs = dict(target0=target0, reward=jnp.float32(-0.1), gamma=jnp.float32(0.99))

        ref_loss = _anchored_reference(theta, h, **kwargs)
        new_loss = _anchored_new(theta, h, **kwargs)
        assert jnp.allclose(ref_loss, new_loss, atol=1e-5)

        ref_grad_theta, ref_grad_h = jax.grad(_anchored_reference, argnums=(0, 1))(theta, h, **kwargs)
        new_grad_theta, new_grad_h = jax.grad(_anchored_new, argnums=(0, 1))(theta, h, **kwargs)
        assert jnp.allclose(ref_grad_theta, new_grad_theta, atol=1e-5)
        assert jnp.allclose(ref_grad_h, new_grad_h, atol=1e-5)


def _discrete_target_free_reference(theta: jax.Array, h: jax.Array, *, reward: jax.Array, gamma: jax.Array) -> jax.Array:
    """discrete_gi_qrc.py's ``_gi_qrc_loss`` target-free branch, before this module existed.

    Already had the duplicate-bootstrap-reference fix (dropping ``h[1]``) baked in, unlike
    gi_sac's original formula -- so, unlike the SAC case above, no amendment is needed here.
    ``theta`` stands in for the shared online value read by both ``q_online`` and
    ``v_next_all`` (own current and own next-value), matching one network's params
    evaluated at two inputs.
    """
    sg = jax.lax.stop_gradient
    q_online = theta
    v_next_all = theta
    v_next_for_head = jnp.concatenate([v_next_all[:1], v_next_all[:-1]])

    y = reward + gamma * v_next_for_head
    delta = sg(y) - q_online
    q_loss = jnp.sum(0.5 * delta**2)

    grad_h = jnp.concatenate([h[:1], h[2:]])
    grad_v_next = jnp.concatenate([v_next_for_head[:1], v_next_for_head[2:]])
    grad_term = jnp.sum(gamma * sg(grad_h) * grad_v_next)
    h_loss = jnp.sum(0.5 * (sg(delta) - h) ** 2)
    return q_loss + grad_term + h_loss


def _discrete_target_free_new(theta: jax.Array, h: jax.Array, *, reward: jax.Array, gamma: jax.Array) -> jax.Array:
    active = jnp.ones_like(h, dtype=bool)
    boot_idx = bootstrap_index(theta.shape[0])
    loss, _ = gtd_iteration_correction(theta, theta, h, active, boot_idx, reward, gamma)
    return loss


class TestMatchesDiscreteGiQrcTargetFreeChain:
    def test_values_and_gradients_match_the_original_formula(self) -> None:
        theta_key, h_key = jax.random.split(jax.random.key(2))
        theta = jax.random.normal(theta_key, (NUM_HEADS,))
        h = jax.random.normal(h_key, (NUM_HEADS,))
        kwargs = dict(reward=jnp.float32(0.4), gamma=jnp.float32(0.9))

        ref_loss = _discrete_target_free_reference(theta, h, **kwargs)
        new_loss = _discrete_target_free_new(theta, h, **kwargs)
        assert jnp.allclose(ref_loss, new_loss, atol=1e-5)

        ref_grad_theta, ref_grad_h = jax.grad(_discrete_target_free_reference, argnums=(0, 1))(theta, h, **kwargs)
        new_grad_theta, new_grad_h = jax.grad(_discrete_target_free_new, argnums=(0, 1))(theta, h, **kwargs)
        assert jnp.allclose(ref_grad_theta, new_grad_theta, atol=1e-5)
        assert jnp.allclose(ref_grad_h, new_grad_h, atol=1e-5)
