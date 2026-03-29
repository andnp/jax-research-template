import time

import jax
from rl_agents.dqn_atari import DQNAtariConfig, dqn_zoo_atari_total_train_env_steps, make_train


def main():
    config = DQNAtariConfig(
        GAME="pong",
        REPLAY_CAPACITY=5_000,
        MIN_REPLAY_CAPACITY_FRACTION=0.2,
        BATCH_SIZE=32,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=4_000,
        NUM_ITERATIONS=1,
        NUM_TRAIN_FRAMES_PER_ITERATION=20_000,
        LEARNING_RATE=1e-4,
        SEED=42,
    )

    rng = jax.random.key(config.SEED)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    print("--- Training DQN on JAXAtari Pong ---")
    print("Compiling & running quick signs-of-life probe...")
    start_time = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    elapsed = time.time() - start_time

    metrics = out["metrics"]
    completed_mask = metrics["returned_episode"].astype(bool)
    returns = metrics["returned_episode_returns"]
    completed_returns = returns[completed_mask]
    env_steps = dqn_zoo_atari_total_train_env_steps(config)
    sps = env_steps / elapsed

    print(f"Elapsed Time:         {elapsed:.2f}s")
    print(f"Env Steps:            {env_steps}")
    print(f"SPS:                  {sps:.2f}")
    print(f"Completed Episodes:   {int(completed_mask.sum().item())}")
    if completed_returns.size:
        print(f"Last Completed Return:{completed_returns[-1].item():.2f}")
        print(f"Max Completed Return: {completed_returns.max().item():.2f}")
    else:
        print("No completed episodes were observed in this short probe.")


if __name__ == "__main__":
    main()