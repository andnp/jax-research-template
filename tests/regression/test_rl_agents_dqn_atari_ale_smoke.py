"""Smoke test: the ported Atari DQN drives the real ALE emulator through the shared loop."""

from __future__ import annotations

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


def test_dqn_atari_ale_smoke() -> None:
    config = DQNAtariConfig(
        replay_capacity=16,
        min_replay_capacity_fraction=0.25,
        batch_size=4,
        learn_period_frames=4,
        target_network_update_period_frames=8,
    )
    runtime_config = dqn_atari_runtime_from_dqn_zoo(
        config,
        num_iterations=1,
        num_train_frames_per_iteration=32,
    )
    steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
    agent = DQNAtariAgent(config, runtime_config)
    env = make_atari_adapter(AleAtariConfig(game="Pong", frame_skip=config.num_action_repeats))

    def train_fn(key: jax.Array):
        return run(agent, env, key, steps=steps, gamma=GAMMA)

    final_state, metrics = jax.jit(train_fn)(jax.random.key(0))

    for key_name in ("loss", "epsilon", "loop/reward", "loop/episode_end", "loop/episode_return"):
        assert metrics[key_name].shape == (steps,), key_name
    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert jnp.any(metrics["loss"] != 0.0), "the replay warmup completed, so the learn path must have fired"
    assert int(final_state.agent_state.replay_state.count) == steps - 1
