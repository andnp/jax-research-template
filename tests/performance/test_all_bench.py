import gymnax
import gymnax.wrappers
import jax
import pytest
from rl_agents.dqn import DQNConfig
from rl_agents.dqn import make_train as make_dqn
from rl_agents.ppo import make_train as make_ppo
from rl_agents.sac import SACConfig
from rl_agents.sac import make_train as make_sac
from rl_components.types import PPOConfig

_ppo_config = PPOConfig(TOTAL_TIMESTEPS=10_000, ENV_NAME="CartPole-v1")
_dqn_config = DQNConfig(TOTAL_TIMESTEPS=10_000, ENV_NAME="CartPole-v1")
_sac_config = SACConfig(TOTAL_TIMESTEPS=10_000, ENV_NAME="MountainCarContinuous-v0")
_ppo_env, _ppo_env_params = gymnax.make(_ppo_config.ENV_NAME)
_ppo_env = gymnax.wrappers.LogWrapper(_ppo_env)
_dqn_env, _dqn_env_params = gymnax.make(_dqn_config.ENV_NAME)
_dqn_env = gymnax.wrappers.LogWrapper(_dqn_env)
_sac_env, _sac_env_params = gymnax.make(_sac_config.ENV_NAME)
_sac_env = gymnax.wrappers.LogWrapper(_sac_env)


@pytest.mark.benchmark(group="ppo")
def test_ppo_speed(benchmark):
    rng = jax.random.PRNGKey(_ppo_config.SEED)
    train_jit = jax.jit(make_ppo(_ppo_config, env=_ppo_env, env_params=_ppo_env_params))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="dqn")
def test_dqn_speed(benchmark):
    rng = jax.random.PRNGKey(_dqn_config.SEED)
    train_jit = jax.jit(make_dqn(_dqn_config, env=_dqn_env, env_params=_dqn_env_params))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=3)


@pytest.mark.benchmark(group="sac")
def test_sac_speed(benchmark):
    rng = jax.random.PRNGKey(_sac_config.SEED)
    train_jit = jax.jit(make_sac(_sac_config, env=_sac_env, env_params=_sac_env_params))
    jax.block_until_ready(train_jit(rng))
    benchmark.pedantic(lambda: jax.block_until_ready(train_jit(rng)), rounds=2)
