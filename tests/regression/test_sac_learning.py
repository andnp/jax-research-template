"""Behavioural gate: SAC must learn Pendulum through the shared training loop.

The threshold is a measured floor, not a target. Pendulum's action range is ``[-2, 2]``
while SAC's tanh policy emits ``[-1, 1]``, so the environment goes through
:class:`~rl_components.action_normalization.ActionNormalizationWrapper` and the agent
commands the full torque range.
"""

import gymnax
import jax
import jax.numpy as jnp
from rl_agents.sac import SACAgent, SACConfig
from rl_components.action_normalization import ActionNormalizationWrapper
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run

GAMMA = 0.99
LATE_EPISODES = 20
"""Completed episodes averaged for the gate, taken from the end of the run."""


def test_sac_pendulum_learns() -> None:
    config = SACConfig(TOTAL_TIMESTEPS=50_000, ENV_NAME="Pendulum-v1", SEED=42)
    raw_env, env_params = gymnax.make(config.ENV_NAME)
    env = ActionNormalizationWrapper(make_gymnax_env(raw_env))
    agent = SACAgent(config)
    train_jit = jax.jit(
        lambda key: run(agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params)
    )
    _, metrics = train_jit(jax.random.key(config.SEED))

    # ``loop/episode_return`` is a sparse impulse: it carries a completed episode's return
    # on the boundary step and zero everywhere else, so it must be masked, not averaged.
    boundaries = jnp.flatnonzero(metrics["loop/episode_end"])
    assert boundaries.size >= LATE_EPISODES, f"only {boundaries.size} episodes completed"
    final_return = metrics["loop/episode_return"][boundaries[-LATE_EPISODES:]].mean()

    print(f"Final mean return: {final_return}")
    assert final_return > -300, f"SAC failed to learn Pendulum. Return: {final_return}"
