"""Behavioural gate: PPO must learn CartPole through the shared training loop."""

import gymnax
import jax
import jax.numpy as jnp
from rl_agents.ppo import PPOAgent, PPOConfig
from rl_components.loop import run
from rl_envs.gymnax_bridge import make_gymnax_env

GAMMA = 0.99
LATE_FRACTION = 0.9
"""Fraction of the horizon after which an episode counts towards the final score."""

_config = PPOConfig(TOTAL_TIMESTEPS=80_000, ENV_NAME="CartPole-v1", NORMALIZE_OBSERVATIONS=True)


def test_ppo_cartpole_learns() -> None:
    raw_env, env_params = gymnax.make(_config.ENV_NAME)
    env = make_gymnax_env(raw_env)
    agent = PPOAgent(_config)

    def train_fn(key: jax.Array):
        return run(
            agent,
            env,
            key,
            steps=_config.TOTAL_TIMESTEPS,
            gamma=GAMMA,
            env_params=env_params,
        )

    train_jit = jax.jit(train_fn)

    _, metrics = train_jit(jax.random.PRNGKey(_config.SEED))

    # ``loop/episode_return`` is a sparse impulse: it carries a completed episode's return
    # on the boundary step and zero everywhere else, so it must be masked, not averaged.
    boundaries = jnp.flatnonzero(metrics["loop/episode_end"])
    late = boundaries[boundaries >= int(LATE_FRACTION * _config.TOTAL_TIMESTEPS)]
    final_return = metrics["loop/episode_return"][late].mean()
    print(f"Episodes: {boundaries.size}, late episodes: {late.size}, final mean return: {final_return}")

    assert late.size > 0, "no episode completed in the final tenth of the run"
    # CartPole-v1 max is 500.
    assert final_return > 400, f"PPO failed to learn CartPole. Return: {final_return}"
