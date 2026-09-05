"""Test GAC on CartPole-v1 via a continuous-action wrapper.

CartPole is discrete (2 actions), but GAC outputs continuous actions
in [-1, 1]. The wrapper maps: negative → 0 (left), positive → 1 (right).
"""

from __future__ import annotations

import time

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rl_agents.greedy_ac import GACAgent, GACConfig
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run

GAMMA = 0.99


class ContinuousCartPole[StateT, ParamsT]:
    """Wraps a gymnax-bridged CartPole-v1 :class:`EnvProtocol` to accept actions in [-1, 1].

    Maps: negative action → discrete action 0 (left),
          positive action → discrete action 1 (right).
    """

    def __init__(self, env: EnvProtocol[jax.Array, StateT, jax.Array, ParamsT]) -> None:
        self._env = env

    def spec(self, params: ParamsT | None = None) -> EnvSpec:
        base = self._env.spec(params)
        return EnvSpec(
            id=base.id,
            observation_shape=base.observation_shape,
            action_shape=(1,),
            observation_dtype=base.observation_dtype,
            action_dtype=jnp.float32,
            action_low=jnp.full((1,), -1.0, dtype=jnp.float32),
            action_high=jnp.full((1,), 1.0, dtype=jnp.float32),
            truncation_policy=base.truncation_policy,
        )

    def reset(self, key: jax.Array, params: ParamsT | None = None) -> EnvReset[jax.Array, StateT]:
        return self._env.reset(key, params)

    def step(self, key: jax.Array, state: StateT, action: jax.Array, params: ParamsT | None = None) -> EnvStep[jax.Array, StateT]:
        # Map [-1, 1] → {0, 1}
        discrete_action = jnp.where(action[0] < 0, 0, 1).astype(jnp.int32)
        return self._env.step(key, state, discrete_action, params)


def main() -> None:
    config = GACConfig(
        ENV_NAME="CartPole-v1",
        TOTAL_TIMESTEPS=100_000,
        BUFFER_SIZE=50_000,
        BATCH_SIZE=32,
        LEARNING_STARTS=1,
        LEARNING_RATE=3e-4,
        ACTOR_LEARNING_RATE=3e-4,
        NUM_SAMPLES=32,
        ACTOR_PERCENTILE=0.1,
        UNIFORM_WEIGHT=0.2,
        ENTROPY_WEIGHT=0.01,
        NUM_RAND_ACTIONS=10,
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = ContinuousCartPole(make_gymnax_env(raw_env))
    agent = GACAgent(config)
    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)

    train_jit = jax.jit(train_fn)

    print(f"--- Training GAC on {config.ENV_NAME} (continuous wrapper) ---")
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
    plt.savefig("gac_cartpole_results.png")
    print("Plot saved as gac_cartpole_results.png")


if __name__ == "__main__":
    main()
