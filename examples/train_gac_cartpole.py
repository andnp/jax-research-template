"""Test GAC on CartPole-v1 via a continuous-action wrapper.

CartPole is discrete (2 actions), but GAC outputs continuous actions
in [-1, 1]. The wrapper maps: negative → 0 (left), positive → 1 (right).
"""

from __future__ import annotations

import time
from typing import Any

import gymnax
import gymnax.wrappers
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rl_agents.greedy_ac import GACConfig, make_train
from rl_components.gym_env import ContinuousActionSpace


class ContinuousCartPole:
    """Wraps gymnax CartPole-v1 to accept continuous actions in [-1, 1].

    Maps: negative action → discrete action 0 (left),
          positive action → discrete action 1 (right).

    Preserves the standard gymnax GymEnv protocol that make_train expects,
    but reports ``action_space`` with a continuous shape so the agent
    outputs a 1D continuous action.
    """

    def __init__(self, env: Any, env_params: Any) -> None:
        self._env = env
        self._env_params = env_params

    def observation_space(self, params: Any = None) -> Any:
        return self._env.observation_space(params or self._env_params)

    def action_space(self, params: Any = None) -> ContinuousActionSpace:
        # Report 1D continuous action space in [-1, 1]
        class _ContActionSpace:
            shape = (1,)
        return _ContActionSpace()  # type: ignore[return-value]

    def reset(self, key: jax.Array, params: Any = None) -> tuple[jax.Array, Any]:
        return self._env.reset(key, params or self._env_params)

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: Any = None,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, jax.Array]]:
        # Map [-1, 1] → {0, 1}
        discrete_action = jnp.where(action[0] < 0, 0, 1).astype(jnp.int32)
        return self._env.step(key, state, discrete_action, params or self._env_params)


def main() -> None:
    config = GACConfig(
        ENV_NAME="CartPole-v1",
        TOTAL_TIMESTEPS=100_000,
        BUFFER_SIZE=50_000,
        BATCH_SIZE=32,
        LEARNING_STARTS=1,
        LR=3e-4,
        ACTOR_LR=3e-4,
        NUM_SAMPLES=32,
        ACTOR_PERCENTILE=0.1,
        UNIFORM_WEIGHT=0.2,
        ENTROPY_WEIGHT=0.01,
        NUM_RAND_ACTIONS=10,
        GAMMA=0.99,
        SEED=42,
    )

    rng = jax.random.PRNGKey(config.SEED)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = ContinuousCartPole(raw_env, env_params)
    env = gymnax.wrappers.LogWrapper(env)

    train_fn = make_train(config, env=env, env_params=env_params)  # type: ignore[arg-type]
    train_jit = jax.jit(train_fn)

    print(f"--- Training GAC on {config.ENV_NAME} (continuous wrapper) ---")
    start = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    elapsed = time.time() - start

    metrics = out["metrics"]
    returns = metrics["returned_episode_returns"]
    sps = config.TOTAL_TIMESTEPS / elapsed
    print(f"Final Return:     {returns[-1].item():.2f}")
    print(f"Max Return:       {returns.max().item():.2f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(returns)
    plt.xlabel("Step")
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
