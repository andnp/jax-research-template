import jax
import pytest

from rl_agents.dqn import DQNConfig
from rl_agents.dqn import make_train as make_dqn
from rl_agents.ppo import make_train as make_ppo
from rl_agents.sac import SACConfig
from rl_agents.sac import make_train as make_sac
from rl_components.types import PPOConfig


@pytest.mark.benchmark(group="ppo")
def test_ppo_speed(benchmark):
    rng = jax.random.PRNGKey(config.SEED)
    train_jit = jax.jit(make_ppo(config))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="dqn")
def test_dqn_speed(benchmark):
    rng = jax.random.PRNGKey(config.SEED)
    train_jit = jax.jit(make_dqn(config))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="sac")
def test_sac_speed(benchmark):
    rng = jax.random.PRNGKey(config.SEED)
    train_jit = jax.jit(make_sac(config))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=2)
