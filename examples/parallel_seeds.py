import time

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rl_agents.ppo import PPOAgent
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run
from rl_components.types import PPOConfig

GAMMA = 0.99


def main() -> None:
    # Philosophical constraint: NUM_ENVS is now implicitly 1 in the agent logic.
    # We scale by running multiple independent experiments (seeds) in parallel.
    NUM_SEEDS = 16

    config = PPOConfig(
        TOTAL_TIMESTEPS=200_000,
        NUM_STEPS=64,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        LEARNING_RATE=3e-4,
        ENV_NAME="CartPole-v1",
    )

    # Each seed gets its own PRNGKey
    rng = jax.random.PRNGKey(config.SEED)
    rng_seeds = jax.random.split(rng, NUM_SEEDS)

    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = PPOAgent(config)

    # VMAP over the seeds!
    # This runs NUM_SEEDS independent agents in parallel.
    parallel_train = jax.vmap(
        lambda key: run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)
    )
    parallel_train = jax.jit(parallel_train)

    print(f"--- Running {NUM_SEEDS} independent PPO agents in parallel ---")
    print("Compiling & Executing...")

    start_time = time.time()
    _, metrics = parallel_train(rng_seeds)
    jax.block_until_ready(metrics)
    total_time = time.time() - start_time

    # returns/ends shape is (NUM_SEEDS, steps)
    episode_returns = metrics["loop/episode_return"]
    episode_ends = metrics["loop/episode_end"]

    sps = (config.TOTAL_TIMESTEPS * NUM_SEEDS) / total_time

    print(f"Total Time:      {total_time:.2f}s")
    print(f"Combined SPS:    {sps:.2f}")

    final_returns = jnp.array(
        [returns[ends][-1] for returns, ends in zip(episode_returns, episode_ends, strict=True)]
    )
    print(f"Mean Final Ret:  {final_returns.mean().item():.2f}")
    print(f"Std Final Ret:   {final_returns.std().item():.2f}")

    # Plotting all seeds
    plt.figure(figsize=(10, 6))
    for seed_index in range(NUM_SEEDS):
        returns = episode_returns[seed_index][episode_ends[seed_index]]
        plt.plot(returns, alpha=0.3, color="blue")

    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"PPO Parallel Seeds (N={NUM_SEEDS}, Total SPS: {sps:.0f})")
    plt.savefig("parallel_ppo_results.png")
    print("Plot saved as parallel_ppo_results.png")


if __name__ == "__main__":
    main()
