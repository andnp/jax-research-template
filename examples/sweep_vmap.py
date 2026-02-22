import jax
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig
import time
import pandas as pd

def sweep_vmap_ppo():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    steps_per_agent = 50_000
    
    config = PPOConfig(
        TOTAL_TIMESTEPS=steps_per_agent,
        ENV_NAME="CartPole-v1",
    )
    
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    
    results = []
    
    print(f"--- Sweeping VMAP Batch Sizes (Steps per agent: {steps_per_agent}) ---")
    
    for b in batch_sizes:
        parallel_train = jax.vmap(train_jit)
        keys = jax.random.split(jax.random.PRNGKey(0), b)
        
        # Warmup
        print(f"Batch Size {b}: Compiling...", end="", flush=True)
        jax.block_until_ready(parallel_train(keys))
        print(" Done. Timing...", end="", flush=True)
        
        start = time.time()
        jax.block_until_ready(parallel_train(keys))
        elapsed = time.time() - start
        
        combined_sps = (steps_per_agent * b) / elapsed
        sps_per_agent = combined_sps / b
        
        print(f" SPS: {combined_sps:.0f} ({sps_per_agent:.0f} per agent)")
        
        results.append({
            "batch_size": b,
            "total_sps": combined_sps,
            "sps_per_agent": sps_per_agent,
            "time": elapsed
        })
    
    df = pd.DataFrame(results)
    df.to_csv("vmap_sweep_results.csv", index=False)
    print("
Results saved to vmap_sweep_results.csv")

if __name__ == "__main__":
    sweep_vmap_ppo()
