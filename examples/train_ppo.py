import time

import jax
import matplotlib.pyplot as plt

from rl_agents.ppo import make_train
from rl_components.types import PPOConfig


def main():
    config = PPOConfig(
        TOTAL_TIMESTEPS=500_000,
        NUM_STEPS=64,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        LR=3e-4,
        ENV_NAME="CartPole-v1",
    )

    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    print(f"--- Training PPO on {config.ENV_NAME} ---")
    print("Compiling...")
    compile_start = time.time()
    # Trigger compilation with a small run or just run it.
    # Here we just run the full thing.
    out = train_jit(rng)
    jax.block_until_ready(out)
    total_time = time.time() - compile_start

    # We want to separate compile time from execution time for real SPS
    # But JAX JIT usually happens on the first call.
    # To get "pure" SPS, we'd run once and then time a second run.
    # For a CLI script, the first run's total time (including JIT) is often what the user cares about.
    # Let's do a quick second run for accurate SPS if it's fast enough.

    print("Executing second run for accurate SPS...")
    start_time = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    execution_time = time.time() - start_time

    metrics = out["metrics"]
    returns = metrics["returned_episode_returns"]
    sps = config.TOTAL_TIMESTEPS / execution_time

    print(f"Compilation Time: {max(0, total_time - execution_time):.2f}s")
    print(f"Execution Time:   {execution_time:.2f}s")
    print(f"SPS:              {sps:.2f}")
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.plot(returns)
    plt.xlabel("Update Step")
    plt.ylabel("Episode Return")
    plt.title(f"PPO on {config.ENV_NAME} (SPS: {sps:.0f})")
    plt.savefig("ppo_results.png")
    print("Plot saved as ppo_results.png")


if __name__ == "__main__":
    main()
