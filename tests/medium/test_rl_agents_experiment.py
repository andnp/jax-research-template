"""rl_train runs a built agent in a built environment, batched over seeds and traced values."""

from typing import cast

import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNConfig
from rl_agents.experiment import RLConfig, rl_train
from rl_envs.env_config import GymnaxConfig

_STEPS = 40
_ENV = GymnaxConfig("CartPole-v1")


def _dqn(learning_rate: float = 3e-4) -> DQNConfig:
    return DQNConfig(
        learning_rate=learning_rate,
        BUFFER_SIZE=64,
        BATCH_SIZE=8,
        LEARNING_STARTS=8,
        TARGET_NETWORK_FREQUENCY=10,
        total_timesteps=_STEPS,
    )


def _train_default(key: jax.Array):
    return rl_train(RLConfig(agent=_dqn(), env=_ENV), key)


def test_rl_train_returns_the_loop_metrics_for_each_seed() -> None:
    keys = jax.random.split(jax.random.key(0), 2)

    _, metrics = jax.jit(jax.vmap(_train_default))(keys)

    assert metrics["loop/reward"].shape == (2, _STEPS)


def test_traced_fields_vary_across_a_vmapped_batch() -> None:
    keys = jax.random.split(jax.random.key(0), 2)
    gammas = jnp.asarray([0.5, 0.99])

    def train(gamma: jax.Array, key: jax.Array):
        traced_gamma = cast(float, gamma)
        agent = _dqn(cast(float, gamma * 1e-3))
        return rl_train(RLConfig(agent=agent, env=_ENV, gamma=traced_gamma), key)

    _, metrics = jax.jit(jax.vmap(train))(gammas, keys)

    discount = metrics["loop/discount"]
    assert discount.shape == (2, _STEPS)
    for row, gamma in zip(discount, gammas, strict=True):
        bootstrapping = row[row != 0]
        assert bootstrapping.size > 0
        assert jnp.allclose(bootstrapping, gamma)
