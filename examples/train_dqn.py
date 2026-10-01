import time

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_components.loop import run
from rl_envs.gymnax_bridge import make_gymnax_env

GAMMA = 0.99


def main() -> None:
    config = DQNConfig(
        total_timesteps=100_000,
        buffer_size=50_000,
        batch_size=128,
        learning_starts=1000,
        TARGET_NETWORK_FREQUENCY=500,
        learning_rate=1e-3,
        ENV_NAME="CartPole-v1",
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = DQNAgent(config)
    def train_fn(key: jax.Array):
        return run(
            agent,
            env,
            key,
            steps=config.total_timesteps,
            gamma=GAMMA,
            env_params=env_params,
        )

    train_jit = jax.jit(train_fn)

    print(f"--- Training DQN on {config.ENV_NAME} ---")
    start = time.time()
    _, metrics = jax.block_until_ready(train_jit(rng))
    elapsed = time.time() - start

    returns = metrics["loop/episode_return"][metrics["loop/episode_end"]]
    sps = config.total_timesteps / elapsed
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
