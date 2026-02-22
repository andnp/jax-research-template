import jax
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig
import time
import argparse
import os

def profile_ppo(steps=1000):
    config = PPOConfig(
        TOTAL_TIMESTEPS=steps,
        NUM_STEPS=64,
        ENV_NAME="CartPole-v1",
    )
    
    rng = jax.random.PRNGKey(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    
    print("Warmup (compiling)...")
    jax.block_until_ready(train_jit(rng))
    
    print(f"Profiling {steps} steps...")
    # Use jax.profiler to start a trace
    # The trace will be saved to the logdir
    logdir = "./trace"
    os.makedirs(logdir, exist_ok=True)
    
    jax.profiler.start_trace(logdir)
    out = train_jit(rng)
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    
    print(f"Trace saved to {logdir}. Use 'tensorboard --logdir={logdir}' to view.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    profile_ppo(args.steps)
