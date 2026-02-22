import jax
import pytest

from agents.dqn import DQNConfig
from agents.dqn import make_train as make_dqn
from agents.ppo import make_train as make_ppo
from agents.sac import SACConfig
from agents.sac import make_train as make_sac
from components.types import PPOConfig


@pytest.mark.benchmark(group="ppo")
def test_ppo_speed(benchmark):
    config = PPOConfig(TOTAL_TIMESTEPS=100_000, NUM_ENVS=32, ENV_NAME="CartPole-v1")
    rng = jax.random.PRNGKey(config.SEED)
    train_jit = jax.jit(make_ppo(config))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="dqn")
def test_dqn_speed(benchmark):
    config = DQNConfig(TOTAL_TIMESTEPS=50_000, NUM_ENVS=4, ENV_NAME="CartPole-v1")
    rng = jax.random.PRNGKey(config.SEED)
    train_jit = jax.jit(make_dqn(config))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="sac")
def test_sac_speed(benchmark):
    config = SACConfig(TOTAL_TIMESTEPS=20_000, NUM_ENVS=1, ENV_NAME="MountainCarContinuous-v0")
    rng = jax.random.PRNGKey(config.SEED)
    train_jit = jax.jit(make_sac(config))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=2)
