import time

import jax
import matplotlib.pyplot as plt

from agents.sac import SACConfig, make_train


def main():
        config = SACConfig(
            TOTAL_TIMESTEPS=100_000,
            BUFFER_SIZE=100_000,
            BATCH_SIZE=256,
            LEARNING_STARTS=1000,
            LR=1e-3,
            ENV_NAME="MountainCarContinuous-v0",
            SEED=42
        )
        
        rng = jax.random.PRNGKey(config.SEED)
        train_fn = make_train(config)
        train_jit = jax.jit(train_fn)
        
        print(f"--- Training SAC on {config.ENV_NAME} ---")
        print("Compiling & Training (1st run)...")
        compile_start = time.time()
        out = train_jit(rng)
        jax.block_until_ready(out)
        total_time = time.time() - compile_start
        
        # SAC is slower, but we still want a pure execution time if possible.
        # For a 50k run, another 20s might be fine.
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
    plt.xlabel("Step")
    plt.ylabel("Episode Return")
    plt.title(f"SAC on {config.ENV_NAME} (SPS: {sps:.0f})")
    plt.savefig("sac_results.png")
    print("Plot saved as sac_results.png")


if __name__ == "__main__":
    main()
