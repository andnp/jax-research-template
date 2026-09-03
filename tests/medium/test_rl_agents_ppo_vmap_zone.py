"""One compiled kernel must train a batch of PPO hyperparameter arms independently."""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from rl_agents.ppo import make_train, ppo_hypers
from rl_components.types import PPOConfig


@dataclass(frozen=True)
class _ObsSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype


@dataclass(frozen=True)
class _ActionSpace:
    n: int


class _TinyEnv:
    """A trivial discrete env with just enough signal to produce nonzero gradients."""

    def observation_space(self, params: object | None = None) -> _ObsSpace:
        del params
        return _ObsSpace(shape=(2,), dtype=jnp.dtype(jnp.float32))

    def action_space(self, params: object | None = None) -> _ActionSpace:
        del params
        return _ActionSpace(n=2)

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del params
        return jax.random.normal(key, (2,), dtype=jnp.float32), jnp.array(0, dtype=jnp.int32)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del state, action, params
        obsv = jax.random.normal(key, (2,), dtype=jnp.float32)
        info = {
            "returned_episode": jnp.array(False),
            "returned_episode_returns": jnp.array(1.0, dtype=jnp.float32),
        }
        return obsv, jnp.array(0, dtype=jnp.int32), jnp.array(1.0, dtype=jnp.float32), jnp.array(False), info


def _tiny_config() -> PPOConfig:
    return PPOConfig(NUM_STEPS=4, TOTAL_TIMESTEPS=8, UPDATE_EPOCHS=1, NUM_MINIBATCHES=1)


def _run_arm(learning_rate: jax.Array, key: jax.Array):
    config = _tiny_config()
    train = make_train(config, env=_TinyEnv(), env_params=None)
    hypers = dataclasses.replace(ppo_hypers(config), LR=learning_rate)
    out = train(key, hypers)
    return out["runner_state"].train_state.params


class TestVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own weights.

        Both arms share one key, so identical weights would mean the swept rate
        never reached the kernel -- the failure that existed while PPOConfig's
        dynamic fields were closed over the outer ``make_train`` closure.
        """
        learning_rates = jnp.array([1e-1, 1e-4], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(_run_arm))(learning_rates, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == learning_rates.shape[0]

        fast, slow = leaves[0][0], leaves[0][1]
        assert not jnp.allclose(fast, slow)

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same rate and key must land on identical weights.

        Guards the negative direction: divergence has to come from the swept
        value, not from the batching itself.
        """
        learning_rates = jnp.array([1e-3, 1e-3], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(_run_arm))(learning_rates, keys)

        leaf = jax.tree_util.tree_leaves(params)[0]
        assert jnp.allclose(leaf[0], leaf[1])
