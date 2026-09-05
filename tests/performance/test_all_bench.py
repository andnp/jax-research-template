from typing import Any

import gymnax
import jax
import pytest
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.ppo import PPOAgent
from rl_agents.sac import SACAgent, SACConfig
from rl_components.loop import run
from rl_components.types import PPOConfig
from rl_envs.gymnax_bridge import make_gymnax_env

GAMMA = 0.99

_ppo_config = PPOConfig(TOTAL_TIMESTEPS=10_000, ENV_NAME="CartPole-v1")
_dqn_config = DQNConfig(TOTAL_TIMESTEPS=10_000, ENV_NAME="CartPole-v1")
_sac_config = SACConfig(TOTAL_TIMESTEPS=10_000, ENV_NAME="MountainCarContinuous-v0")
_ppo_raw_env, _ppo_env_params = gymnax.make(_ppo_config.ENV_NAME)
_ppo_env = make_gymnax_env(_ppo_raw_env)
_dqn_raw_env, _dqn_env_params = gymnax.make(_dqn_config.ENV_NAME)
_dqn_env = make_gymnax_env(_dqn_raw_env)
_sac_raw_env, _sac_env_params = gymnax.make(_sac_config.ENV_NAME)
_sac_env = make_gymnax_env(_sac_raw_env)


@pytest.mark.benchmark(group="ppo")
def test_ppo_speed(benchmark: Any) -> None:
    rng = jax.random.PRNGKey(_ppo_config.SEED)
    agent = PPOAgent(_ppo_config)

    def train_fn(key: jax.Array):
        return run(
            agent,
            _ppo_env,
            key,
            steps=_ppo_config.TOTAL_TIMESTEPS,
            gamma=GAMMA,
            env_params=_ppo_env_params,
        )

    train_jit = jax.jit(train_fn)
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="dqn")
def test_dqn_speed(benchmark: Any) -> None:
    rng = jax.random.PRNGKey(_dqn_config.SEED)
    agent = DQNAgent(_dqn_config)

    def train_fn(key: jax.Array):
        return run(
            agent,
            _dqn_env,
            key,
            steps=_dqn_config.TOTAL_TIMESTEPS,
            gamma=GAMMA,
            env_params=_dqn_env_params,
        )

    train_jit = jax.jit(train_fn)
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="sac")
def test_sac_speed(benchmark: Any) -> None:
    rng = jax.random.PRNGKey(_sac_config.SEED)
    agent = SACAgent(_sac_config)

    def train_fn(key: jax.Array):
        return run(
            agent,
            _sac_env,
            key,
            steps=_sac_config.TOTAL_TIMESTEPS,
            gamma=GAMMA,
            env_params=_sac_env_params,
        )

    train_jit = jax.jit(train_fn)
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=2)
