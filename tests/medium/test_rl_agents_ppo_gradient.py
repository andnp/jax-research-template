"""Medium tests for rl_agents.ppo — gradient flow and JIT compilation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import rl_agents.ppo as ppo_module
from flax.training.train_state import TrainState
from rl_agents.ppo import make_train
from rl_components.networks import ActorCritic
from rl_components.types import PPOConfig


@dataclass(frozen=True)
class FakeObservationSpace:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class FakeActionSpace:
    n: int


class FakeDiscreteEnv:
    def observation_space(self, params: object | None = None) -> FakeObservationSpace:
        del params
        return FakeObservationSpace(shape=(4,))

    def action_space(self, params: object | None = None) -> FakeActionSpace:
        del params
        return FakeActionSpace(n=2)

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del key, params
        return jnp.zeros((4,), dtype=jnp.float32), jnp.array(0, dtype=jnp.int32)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        info = {
            "returned_episode": jnp.array(False),
            "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
        }
        return (
            jnp.full((4,), next_state, dtype=jnp.float32),
            next_state,
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array(False),
            info,
        )


class TestPPOGradientFlow:
    def test_make_train_accepts_injected_env(self) -> None:
        config = PPOConfig(TOTAL_TIMESTEPS=4, NUM_STEPS=2, UPDATE_EPOCHS=1, NUM_MINIBATCHES=1)
        train = make_train(config, env=FakeDiscreteEnv(), env_params=None)

        out = jax.jit(train)(jax.random.key(0))
        metrics = out["metrics"]

        assert metrics["returned_episode"].shape == (2,)
        assert metrics["returned_episode_returns"].shape == (2,)

    def test_make_train_keeps_default_gymnax_resolution(self, monkeypatch) -> None:
        config = PPOConfig(ENV_NAME="FakeDiscrete-v0", TOTAL_TIMESTEPS=4, NUM_STEPS=2, UPDATE_EPOCHS=1, NUM_MINIBATCHES=1)

        def fake_make(env_name: str) -> tuple[FakeDiscreteEnv, object | None]:
            assert env_name == "FakeDiscrete-v0"
            return FakeDiscreteEnv(), None

        monkeypatch.setattr(ppo_module.gymnax, "make", fake_make)
        monkeypatch.setattr(ppo_module.gymnax.wrappers, "LogWrapper", lambda env: env)

        out = jax.jit(make_train(config))(jax.random.key(0))
        metrics = out["metrics"]

        assert metrics["returned_episode"].shape == (2,)
        assert metrics["returned_episode_returns"].shape == (2,)

    def test_params_change_after_update(self):
        """ActorCritic params should change after a PPO gradient step."""
        net = ActorCritic(action_dim=2)
        key = jax.random.key(0)
        params = net.init(key, jnp.zeros((4,)))

        cfg = PPOConfig(CLIP_EPS=0.2, VF_COEF=0.5, ENT_COEF=0.01, MAX_GRAD_NORM=0.5)
        tx = optax.chain(optax.clip_by_global_norm(cfg.MAX_GRAD_NORM), optax.adam(cfg.LR, eps=1e-5))
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        # Fake trajectory batch
        batch_size = 16
        obs = jax.random.normal(jax.random.key(1), (batch_size, 4))
        actions = jax.random.randint(jax.random.key(2), (batch_size,), 0, 2)
        old_log_probs = -jnp.ones((batch_size,)) * 0.5
        advantages = jax.random.normal(jax.random.key(3), (batch_size,))
        targets = jax.random.normal(jax.random.key(4), (batch_size,))

        def loss_fn(params):
            probs, value = net.apply(params, obs)
            log_prob = probs.log_prob(actions)

            # Value loss
            value_loss = 0.5 * jnp.mean(jnp.square(value - targets))

            # Policy loss (clipped)
            ratio = jnp.exp(log_prob - old_log_probs)
            gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss1 = ratio * gae
            loss2 = jnp.clip(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * gae
            policy_loss = -jnp.minimum(loss1, loss2).mean()

            # Entropy loss
            entropy_loss = probs.entropy().mean()

            return policy_loss + cfg.VF_COEF * value_loss - cfg.ENT_COEF * entropy_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(train_state.params)
        new_state = train_state.apply_gradients(grads=grads)

        old_flat = jax.tree_util.tree_leaves(train_state.params)
        new_flat = jax.tree_util.tree_leaves(new_state.params)
        any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=True))
        assert any_changed, "Parameters did not change after gradient step"

    def test_ppo_loss_jit(self):
        """PPO loss should JIT-compile without errors."""
        net = ActorCritic(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))

        @jax.jit
        def compute_loss(params, obs, actions, old_log_probs, advantages, targets):
            probs, value = net.apply(params, obs)
            log_prob = probs.log_prob(actions)
            ratio = jnp.exp(log_prob - old_log_probs)
            gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss1 = ratio * gae
            loss2 = jnp.clip(ratio, 0.8, 1.2) * gae
            policy_loss = -jnp.minimum(loss1, loss2).mean()
            value_loss = 0.5 * jnp.mean(jnp.square(value - targets))
            entropy_loss = probs.entropy().mean()
            return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

        loss = compute_loss(
            params,
            jnp.ones((8, 4)),
            jnp.zeros((8,), dtype=jnp.int32),
            -jnp.ones((8,)) * 0.5,
            jnp.ones((8,)),
            jnp.ones((8,)),
        )
        assert loss.shape == ()
