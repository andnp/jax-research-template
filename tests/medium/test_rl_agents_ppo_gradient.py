"""Medium tests for rl_agents.ppo — gradient flow and JIT compilation."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_agents.ppo import PPOTransition, _sum_action_event_terms, ppo_loss
from rl_components.networks import ActorCritic, ContinuousActorCritic

_CLIP_EPS = jnp.asarray(0.2, dtype=jnp.float32)
_VF_COEF = jnp.asarray(0.5, dtype=jnp.float32)
_ENT_COEF = jnp.asarray(0.01, dtype=jnp.float32)


def _minibatch(
    obs: jax.Array,
    action: jax.Array,
    log_prob: jax.Array,
    value: jax.Array,
    valid: jax.Array,
) -> PPOTransition:
    """Build a minibatch of rollout slots; `ppo_loss` reads only these five fields.

    The reward, discount, episode_end and bootstrap_value a slot also carries are consumed
    upstream by `ppo_gae`, so they stay zero here rather than being given decorative values
    that a reader might mistake for inputs to the loss.
    """
    length = value.shape[0]
    zeros = jnp.zeros((length,), dtype=jnp.float32)
    return PPOTransition(
        obs=obs,
        action=action,
        log_prob=log_prob,
        value=value,
        reward=zeros,
        discount=zeros,
        episode_end=jnp.zeros((length,), dtype=jnp.bool_),
        bootstrap_value=zeros,
        valid=valid,
    )


def _tree_all_finite(tree: object) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


class TestPPOGradientFlow:
    def test_params_change_after_update(self) -> None:
        """A gradient step on the real `ppo_loss` must move the ActorCritic parameters."""
        net = ActorCritic(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4, eps=1e-5))
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        batch_size = 16
        transitions = _minibatch(
            obs=jax.random.normal(jax.random.key(1), (batch_size, 4)),
            action=jax.random.randint(jax.random.key(2), (batch_size,), 0, 2),
            log_prob=-jnp.ones((batch_size,), dtype=jnp.float32) * 0.5,
            value=jax.random.normal(jax.random.key(5), (batch_size,)),
            valid=jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        advantages = jax.random.normal(jax.random.key(3), (batch_size,))
        targets = jax.random.normal(jax.random.key(4), (batch_size,))

        (loss, components), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            train_state.params,
            transitions,
            advantages,
            targets,
            apply_fn=net.apply,
            is_continuous=False,
            clip_eps=_CLIP_EPS,
            vf_coef=_VF_COEF,
            ent_coef=_ENT_COEF,
        )
        new_state = train_state.apply_gradients(grads=grads)

        assert loss.shape == ()
        assert all(component.shape == () for component in components)
        old_flat = jax.tree_util.tree_leaves(train_state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed, "Parameters did not change after gradient step"

    def test_invalid_slots_cannot_move_the_loss(self) -> None:
        """Every component takes a masked mean, so an invalid slot's contents are inert.

        The run's one invalid slot -- global `t = 0`, which closes no transition -- carries
        zero-primed junk in every field. Replacing that junk with wildly different numbers
        must leave all three loss components bit-identical, which fails any component that
        reverted to a plain `.mean()`.
        """
        net = ActorCritic(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))

        batch_size = 8
        valid = jnp.arange(batch_size) != 0
        obs = jax.random.normal(jax.random.key(1), (batch_size, 4))
        action = jax.random.randint(jax.random.key(2), (batch_size,), 0, 2)
        log_prob = -jnp.ones((batch_size,), dtype=jnp.float32) * 0.5
        value = jnp.zeros((batch_size,), dtype=jnp.float32)
        advantages = jax.random.normal(jax.random.key(3), (batch_size,))
        targets = jax.random.normal(jax.random.key(4), (batch_size,))

        def loss_of(
            transitions: PPOTransition, advantages: jax.Array, targets: jax.Array
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
            return ppo_loss(
                params,
                transitions,
                advantages,
                targets,
                apply_fn=net.apply,
                is_continuous=False,
                clip_eps=_CLIP_EPS,
                vf_coef=_VF_COEF,
                ent_coef=_ENT_COEF,
            )

        clean, clean_components = loss_of(_minibatch(obs, action, log_prob, value, valid), advantages, targets)
        perturbed, perturbed_components = loss_of(
            _minibatch(
                obs.at[0].set(400.0),
                action.at[0].set(1 - action[0]),
                log_prob.at[0].set(-9.0),
                value.at[0].set(-70.0),
                valid,
            ),
            advantages.at[0].set(50.0),
            targets.at[0].set(-30.0),
        )

        # The advantages are standardized across the whole minibatch, which the perturbed
        # slot does move, so the policy term legitimately shifts. The value and entropy
        # terms read the slot only through the mask.
        assert jnp.array_equal(perturbed_components[0], clean_components[0]), "value loss read an invalid slot"
        assert jnp.array_equal(perturbed_components[2], clean_components[2]), "entropy read an invalid slot"
        # The perturbation has to reach the loss somewhere, or the assertions are vacuous.
        assert not jnp.array_equal(perturbed, clean)

    def test_ppo_loss_jit(self) -> None:
        """The real `ppo_loss` must JIT-compile and stay finite."""
        net = ActorCritic(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        transitions = _minibatch(
            obs=jnp.ones((8, 4), dtype=jnp.float32),
            action=jnp.zeros((8,), dtype=jnp.int32),
            log_prob=-jnp.ones((8,), dtype=jnp.float32) * 0.5,
            value=jnp.ones((8,), dtype=jnp.float32),
            valid=jnp.ones((8,), dtype=jnp.bool_),
        )

        loss, components = jax.jit(ppo_loss, static_argnames=("apply_fn", "is_continuous"))(
            params,
            transitions,
            jnp.ones((8,), dtype=jnp.float32),
            jnp.ones((8,), dtype=jnp.float32),
            apply_fn=net.apply,
            is_continuous=False,
            clip_eps=_CLIP_EPS,
            vf_coef=_VF_COEF,
            ent_coef=_ENT_COEF,
        )

        assert loss.shape == ()
        assert _tree_all_finite((loss, components))


class TestContinuousPPOGradientFlow:
    def test_params_change_after_update(self) -> None:
        """A gradient step on the real `ppo_loss` must move the continuous head too.

        The continuous path differs only in that `log_prob` and `entropy` come back per
        action dimension and have to be summed, which `is_continuous=True` selects.
        """
        net = ContinuousActorCritic(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4, eps=1e-5))
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        batch_size = 16
        obs = jax.random.normal(jax.random.key(1), (batch_size, 4))
        policy, _ = net.apply(train_state.params, obs)
        actions = policy.sample(seed=jax.random.key(2))
        transitions = _minibatch(
            obs=obs,
            action=actions,
            log_prob=_sum_action_event_terms(policy.log_prob(actions), is_continuous=True),
            value=jax.random.normal(jax.random.key(5), (batch_size,)),
            valid=jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        advantages = jax.random.normal(jax.random.key(3), (batch_size,))
        targets = jax.random.normal(jax.random.key(4), (batch_size,))

        (loss, components), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            train_state.params,
            transitions,
            advantages,
            targets,
            apply_fn=net.apply,
            is_continuous=True,
            clip_eps=_CLIP_EPS,
            vf_coef=_VF_COEF,
            ent_coef=_ENT_COEF,
        )
        new_state = train_state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(train_state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert loss.shape == ()
        assert all(component.shape == () for component in components)
        assert any_changed, "Parameters did not change after continuous PPO gradient step"

    def test_continuous_ppo_loss_jit(self) -> None:
        """The real `ppo_loss` must JIT-compile on the continuous path as well."""
        net = ContinuousActorCritic(action_dim=2)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        transitions = _minibatch(
            obs=jnp.ones((8, 4), dtype=jnp.float32),
            action=jnp.zeros((8, 2), dtype=jnp.float32),
            log_prob=-jnp.ones((8,), dtype=jnp.float32) * 0.5,
            value=jnp.ones((8,), dtype=jnp.float32),
            valid=jnp.ones((8,), dtype=jnp.bool_),
        )

        loss, components = jax.jit(ppo_loss, static_argnames=("apply_fn", "is_continuous"))(
            params,
            transitions,
            jnp.ones((8,), dtype=jnp.float32),
            jnp.ones((8,), dtype=jnp.float32),
            apply_fn=net.apply,
            is_continuous=True,
            clip_eps=_CLIP_EPS,
            vf_coef=_VF_COEF,
            ent_coef=_ENT_COEF,
        )

        assert loss.shape == ()
        assert _tree_all_finite((loss, components))
