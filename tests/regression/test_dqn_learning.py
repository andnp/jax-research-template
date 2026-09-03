"""Behavioural gate: DQN must learn CartPole through the shared training loop."""

import gymnax
import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run

GAMMA = 0.99
LATE_FRACTION = 0.9
"""Fraction of the horizon after which an episode counts towards the final score."""


def test_dqn_cartpole_learns() -> None:
    config = DQNConfig(
        ENV_NAME="CartPole-v1",
        TOTAL_TIMESTEPS=50_000,
    )

    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = DQNAgent(config)
    train_jit = jax.jit(
        lambda key: run(
            agent,
            env,
            key,
            steps=config.TOTAL_TIMESTEPS,
            gamma=GAMMA,
            env_params=env_params,
        )
    )

    _, metrics = train_jit(jax.random.PRNGKey(config.SEED))

    # ``loop/episode_return`` is a sparse impulse: it carries a completed episode's return
    # on the boundary step and zero everywhere else, so it must be masked, not averaged.
    boundaries = jnp.flatnonzero(metrics["loop/episode_end"])
    late = boundaries[boundaries >= int(LATE_FRACTION * config.TOTAL_TIMESTEPS)]
    final_return = metrics["loop/episode_return"][late].mean()
    print(f"Episodes: {boundaries.size}, late episodes: {late.size}, final mean return: {final_return}")

    assert late.size > 0, "no episode completed in the final tenth of the run"
    assert final_return > 100, f"DQN failed to learn CartPole. Return: {final_return}"
