import time

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run

GAMMA = 0.99


def main() -> None:
    config = DQNConfig(
        TOTAL_TIMESTEPS=100_000,
        BUFFER_SIZE=50_000,
        BATCH_SIZE=128,
        LEARNING_STARTS=1000,
        TARGET_NETWORK_FREQUENCY=500,
        LR=1e-3,
        ENV_NAME="CartPole-v1",
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = DQNAgent(config)
    train_jit = jax.jit(
        lambda key: run(
            agent,
            env,
            key,
            steps=config.TOTAL_TIMESTEPS,
            gamma=GAMMA,
            env_params=env_params,
        )
    )

    print(f"--- Training DQN on {config.ENV_NAME} ---")
    start = time.time()
    _, metrics = jax.block_until_ready(train_jit(rng))
    elapsed = time.time() - start

    returns = metrics["loop/episode_return"][metrics["loop/episode_end"]]
    sps = config.TOTAL_TIMESTEPS / elapsed
    print(f"Episodes:         {returns.size}")
    print(f"Final Return:     {jnp.mean(returns[-10:]).item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"DQN on {config.ENV_NAME} (SPS: {sps:.0f})")
    plt.savefig("dqn_results.png")
    print("Plot saved as dqn_results.png")


if __name__ == "__main__":
    main()
