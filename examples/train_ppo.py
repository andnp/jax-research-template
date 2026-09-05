import time

import gymnax
import jax
import matplotlib.pyplot as plt
from rl_agents.ppo import PPOAgent
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run
from rl_components.types import PPOConfig

GAMMA = 0.99


def main() -> None:
    config = PPOConfig(
        TOTAL_TIMESTEPS=500_000,
        NUM_STEPS=64,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        LEARNING_RATE=3e-4,
        ENV_NAME="CartPole-v1",
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = PPOAgent(config)
    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)

    train_jit = jax.jit(train_fn)

    print(f"--- Training PPO on {config.ENV_NAME} ---")
    print("Compiling...")
    compile_start = time.time()
    _, metrics = jax.block_until_ready(train_jit(rng))
    total_time = time.time() - compile_start

    print("Executing second run for accurate SPS...")
    start_time = time.time()
    _, metrics = jax.block_until_ready(train_jit(rng))
    execution_time = time.time() - start_time

    returns = metrics["loop/episode_return"][metrics["loop/episode_end"]]
    sps = config.TOTAL_TIMESTEPS / execution_time

    print(f"Compilation Time: {max(0, total_time - execution_time):.2f}s")
    print(f"Execution Time:   {execution_time:.2f}s")
    print(f"SPS:              {sps:.2f}")
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"PPO on {config.ENV_NAME} (SPS: {sps:.0f})")
    plt.savefig("ppo_results.png")
    print("Plot saved as ppo_results.png")


if __name__ == "__main__":
    main()
