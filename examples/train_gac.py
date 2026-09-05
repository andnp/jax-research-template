"""Example: train Greedy Actor-Critic on continuous control environments.

Usage:
    python examples/train_gac.py                           # MountainCarContinuous
    python examples/train_gac.py Pendulum-v1               # Pendulum (needs normalization)

Supports environments with action range [-1, 1]. For environments with
different action bounds, the ``ActionNormalizationWrapper`` from
``rl_components.action_normalization`` can be used.
"""

from __future__ import annotations

import sys
import time

import gymnax
import jax
import matplotlib.pyplot as plt
from rl_agents.greedy_ac import GACAgent, GACConfig
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run

GAMMA = 0.99


def train_gac(
    env_name: str,
    total_timesteps: int = 100_000,
    seed: int = 42,
) -> None:
    """Train GAC and plot results.

    Args:
        env_name: Gymnax environment name (continuous action, [-1, 1] range).
        total_timesteps: Number of environment steps.
        seed: Random seed.
    """
    config = GACConfig(
        ENV_NAME=env_name,
        TOTAL_TIMESTEPS=total_timesteps,
        BUFFER_SIZE=50_000,
        BATCH_SIZE=64,
        LEARNING_STARTS=500,
        LEARNING_RATE=3e-4,
        ACTOR_LEARNING_RATE=3e-4,
        ACTOR_PERCENTILE=0.1,
        ENTROPY_WEIGHT=0.01,
        SEED=seed,
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = GACAgent(config)
    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)

    train_jit = jax.jit(train_fn)

    print(f"--- Training GAC on {config.ENV_NAME} ---")
    start = time.time()
    _, metrics = jax.block_until_ready(train_jit(rng))
    elapsed = time.time() - start

    returns = metrics["loop/episode_return"][metrics["loop/episode_end"]]
    sps = config.TOTAL_TIMESTEPS / elapsed
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"GAC on {config.ENV_NAME} (SPS: {sps:.0f})")

    plt.subplot(1, 2, 2)
    for key in ("critic_loss", "actor_loss"):
        if key in metrics:
            plt.plot(metrics[key], label=key)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()

    plt.tight_layout()
    save_path = f"gac_{config.ENV_NAME.lower().replace('-', '_')}_results.png"
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")


def main() -> None:
    env_name = sys.argv[1] if len(sys.argv) > 1 else "MountainCarContinuous-v0"
    train_gac(env_name=env_name, total_timesteps=50_000, seed=42)


if __name__ == "__main__":
    main()
