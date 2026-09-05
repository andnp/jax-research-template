import time

import gymnax
import jax
import matplotlib.pyplot as plt
from rl_agents.sac import SACAgent, SACConfig
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run

GAMMA = 0.99


def main() -> None:
    config = SACConfig(
        TOTAL_TIMESTEPS=100_000,
        BUFFER_SIZE=100_000,
        BATCH_SIZE=256,
        LEARNING_STARTS=1000,
        LEARNING_RATE=1e-3,
        ENV_NAME="MountainCarContinuous-v0",
        SEED=42,
    )

    rng = jax.random.key(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = SACAgent(config)
    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)

    train_jit = jax.jit(train_fn)

    print(f"--- Training SAC on {config.ENV_NAME} ---")
    start = time.time()
    _, metrics = jax.block_until_ready(train_jit(rng))
    elapsed = time.time() - start

    returns = metrics["loop/episode_return"][metrics["loop/episode_end"]]
    sps = config.TOTAL_TIMESTEPS / elapsed
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"SAC on {config.ENV_NAME} (SPS: {sps:.0f})")
    plt.savefig("sac_results.png")
    print("Plot saved as sac_results.png")


if __name__ == "__main__":
    main()
