import time

import jax
import jax.numpy as jnp
from rl_agents.dqn_atari import (
    DQNAtariAgent,
    DQNAtariConfig,
    dqn_atari_runtime_from_dqn_zoo,
    dqn_zoo_atari_total_train_env_steps,
)
from rl_components.loop import run
from rl_envs.atari_ale import AleAtariConfig, make_atari_adapter

GAMMA = 0.99


def main() -> None:
    config = DQNAtariConfig(
        replay_capacity=5_000,
        min_replay_capacity_fraction=0.2,
        batch_size=32,
        target_network_update_period_frames=4_000,
        learning_rate=1e-4,
    )
    runtime_config = dqn_atari_runtime_from_dqn_zoo(
        config,
        num_iterations=1,
        num_train_frames_per_iteration=20_000,
        seed=42,
    )
    env_steps = dqn_zoo_atari_total_train_env_steps(runtime_config)

    rng = jax.random.key(runtime_config.SEED)
    agent = DQNAtariAgent(config, runtime_config)
    env = make_atari_adapter(
        AleAtariConfig(
            game="Pong",
            frame_skip=config.num_action_repeats,
        )
    )
    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=env_steps, gamma=GAMMA)

    train_jit = jax.jit(train_fn)

    print("--- Training DQN on ALE Pong ---")
    print("Compiling & running quick signs-of-life probe...")
    start_time = time.time()
    _final_state, metrics = train_jit(rng)
    jax.block_until_ready(metrics)
    elapsed = time.time() - start_time

    # loop/episode_return is a sparse impulse: the completed episode's return on a
    # boundary step and zero everywhere else, so it must be masked, never averaged.
    completed_mask = metrics["loop/episode_end"].astype(bool)
    completed_returns = metrics["loop/episode_return"][completed_mask]
    sps = env_steps / elapsed

    print(f"Elapsed Time:         {elapsed:.2f}s")
    print(f"Env Steps:            {env_steps}")
    print(f"SPS:                  {sps:.2f}")
    print(f"Completed Episodes:   {int(completed_mask.sum().item())}")
    print(f"Mean Loss:            {float(jnp.mean(metrics['loss']).item()):.6f}")
    if completed_returns.size:
        print(f"Last Completed Return:{completed_returns[-1].item():.2f}")
        print(f"Max Completed Return: {completed_returns.max().item():.2f}")
    else:
        print("No completed episodes were observed in this short probe.")


if __name__ == "__main__":
    main()
