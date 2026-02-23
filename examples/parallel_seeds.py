import time

import jax
import matplotlib.pyplot as plt
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig


def main():
    # Philosophical constraint: NUM_ENVS is now implicitly 1 in the agent logic.
    # We scale by running multiple independent experiments (seeds) in parallel.
    NUM_SEEDS = 16 
    
    config = PPOConfig(
        TOTAL_TIMESTEPS=200_000,
        NUM_STEPS=64,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        LR=3e-4,
        ENV_NAME="CartPole-v1",
    )
    
    # Each seed gets its own PRNGKey
    rng = jax.random.PRNGKey(config.SEED)
    rng_seeds = jax.random.split(rng, NUM_SEEDS)
    
    train_fn = make_train(config)

    # VMAP over the seeds!
    # This runs NUM_SEEDS independent agents in parallel.
    parallel_train = jax.vmap(jax.jit(train_fn))
    
    print(f"--- Running {NUM_SEEDS} independent PPO agents in parallel ---")
    print("Compiling & Executing...")
    
    start_time = time.time()
    out = parallel_train(rng_seeds)
    jax.block_until_ready(out)
    total_time = time.time() - start_time
    
    metrics = out["metrics"]
    # returns shape is (NUM_SEEDS, num_updates)
    returns = metrics["returned_episode_returns"]
    
    sps = (config.TOTAL_TIMESTEPS * NUM_SEEDS) / total_time
    
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Combined SPS:    {sps:.2f}")
    print(f"Mean Final Ret:  {returns[:, -1].mean().item():.2f}")
    print(f"Std Final Ret:   {returns[:, -1].std().item():.2f}")
    
    # Plotting all seeds
    plt.figure(figsize=(10, 6))
    for i in range(NUM_SEEDS):
        plt.plot(returns[i], alpha=0.3, color='blue')
    
    plt.plot(returns.mean(axis=0), color='red', linewidth=2, label='Mean')
    plt.xlabel("Update Step")
    plt.ylabel("Episode Return")
    plt.title(f"PPO Parallel Seeds (N={NUM_SEEDS}, Total SPS: {sps:.0f})")
    plt.legend()
    plt.savefig("parallel_ppo_results.png")
    print("Plot saved as parallel_ppo_results.png")

if __name__ == "__main__":
    main()
