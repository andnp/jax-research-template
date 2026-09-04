from typing import Any

import gymnax
import jax
from rl_agents.ppo import PPOAgent
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run
from rl_components.types import PPOConfig

GAMMA = 0.99


def test_ppo_speed(benchmark: Any) -> None:
    config = PPOConfig(
        TOTAL_TIMESTEPS=100_000,
        NUM_STEPS=64,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        LR=3e-4,
        ENV_NAME="CartPole-v1",
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = PPOAgent(config)
    train_jit = jax.jit(
        lambda key: run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)
    )

    # Warmup (compilation)
    jax.block_until_ready(train_jit(rng))

    def run_train() -> Any:
        out = train_jit(rng)
        jax.block_until_ready(out)
        return out

    benchmark.pedantic(run_train, rounds=5, warmup_rounds=1)
