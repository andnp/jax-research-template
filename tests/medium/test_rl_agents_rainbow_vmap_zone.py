"""One compiled Rainbow kernel must train a batch of hyperparameter arms independently."""

import dataclasses

import jax
import jax.numpy as jnp
from rl_agents.rainbow import (
    RainbowConfig,
    RainbowHypers,
    make_train,
    rainbow_atari_runtime_from_dqn_zoo,
    rainbow_hypers,
)
from rl_components.gym_env import DiscreteActionSpace, ObservationSpace


@dataclasses.dataclass(frozen=True)
class _FakeObservationSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype


@dataclasses.dataclass(frozen=True)
class _FakeActionSpace:
    n: int


class _FakeAtariLikeEnv:
    """A constant-reward, never-terminating env -- enough to exercise the learn path."""

    def observation_space(self, params: object | None = None) -> ObservationSpace:
        del params
        return _FakeObservationSpace(shape=(4, 84, 84, 1), dtype=jnp.uint8)

    def action_space(self, params: object | None = None) -> DiscreteActionSpace:
        del params
        return _FakeActionSpace(n=3)

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, jax.Array]:
        del key, params
        return jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8), jnp.array(0, dtype=jnp.int32)

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        observation = jnp.full((4, 84, 84, 1), next_state, dtype=jnp.uint8)
        reward = jnp.array(1.0, dtype=jnp.float32)
        done = jnp.array(False)
        info = {
            "returned_episode": jnp.array(False),
            "returned_episode_returns": jnp.array(0.0, dtype=jnp.float32),
        }
        return observation, next_state, reward, done, info


def _config() -> RainbowConfig:
    return RainbowConfig(
        REPLAY_CAPACITY=16,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=4,
        LEARN_PERIOD_FRAMES=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
    )


def _run_arm(config: RainbowConfig, hypers: RainbowHypers, key: jax.Array):
    runtime_config = rainbow_atari_runtime_from_dqn_zoo(config, num_iterations=1, num_train_frames_per_iteration=32)
    train = make_train(config, runtime_config, _FakeAtariLikeEnv(), env_params=None)
    out = train(key, hypers)
    return out["runner_state"].train_state.params


class TestRainbowVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own weights.

        Both arms share one key, so identical weights would mean the swept
        rate never reached the kernel -- the failure that existed while
        ``LEARNING_RATE`` was closed over the ``RainbowConfig`` baked into the
        optimizer at construction.
        """
        config = _config()
        base_hypers = rainbow_hypers(config)
        learning_rates = jnp.array([1e-1, 1e-4], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        def arm(lr: jax.Array, k: jax.Array):
            hypers = dataclasses.replace(base_hypers, LEARNING_RATE=lr)
            return _run_arm(config, hypers, k)

        params = jax.jit(jax.vmap(arm))(learning_rates, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == learning_rates.shape[0]

        fast, slow = leaves[0][0], leaves[0][1]
        assert not jnp.allclose(fast, slow)

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same hypers and key must land on identical weights.

        Guards the negative direction: divergence has to come from the swept
        value, not from the batching itself.
        """
        config = _config()
        base_hypers = rainbow_hypers(config)
        learning_rates = jnp.array([1e-3, 1e-3], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        def arm(lr: jax.Array, k: jax.Array):
            hypers = dataclasses.replace(base_hypers, LEARNING_RATE=lr)
            return _run_arm(config, hypers, k)

        params = jax.jit(jax.vmap(arm))(learning_rates, keys)

        leaf = jax.tree_util.tree_leaves(params)[0]
        assert jnp.allclose(leaf[0], leaf[1])

    def test_batched_discounts_train_independent_arms(self) -> None:
        """Two ``ADDITIONAL_DISCOUNT`` arms sharing a key must diverge too.

        Uses ``N_STEP=1`` so the n-step reward-shaping matrix (which also
        reads the discount, and would mask the bug below on its own) cannot
        contribute any divergence -- ``discount ** reward_offsets`` is always
        ``discount ** 0 == 1`` at that width. The only remaining place
        ``ADDITIONAL_DISCOUNT`` can reach the trained weights is
        ``bootstrap_discount = hypers.ADDITIONAL_DISCOUNT ** N_STEP`` inside
        the categorical loss, which is exponentiated by the static ``N_STEP``
        at trace-setup time -- the one place a sweep could silently share one
        arm's discount if that exponentiation were computed from the
        closed-over config instead of from ``hypers``.
        """
        config = dataclasses.replace(_config(), N_STEP=1)
        base_hypers = rainbow_hypers(config)
        discounts = jnp.array([0.1, 0.99], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        def arm(discount: jax.Array, k: jax.Array):
            hypers = dataclasses.replace(base_hypers, ADDITIONAL_DISCOUNT=discount)
            return _run_arm(config, hypers, k)

        params = jax.jit(jax.vmap(arm))(discounts, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        low, high = leaves[0][0], leaves[0][1]
        assert not jnp.allclose(low, high)
