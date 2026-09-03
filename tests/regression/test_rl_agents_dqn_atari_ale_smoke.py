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
from rl_components.atari_ale import AleAtariConfig, make_atari_adapter
from rl_components.loop import run

GAMMA = 0.99


def test_dqn_atari_ale_smoke() -> None:
    config = DQNAtariConfig(
        REPLAY_CAPACITY=16,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=4,
        LEARN_PERIOD_FRAMES=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
    )
    runtime_config = dqn_atari_runtime_from_dqn_zoo(
        config,
        num_iterations=1,
        num_train_frames_per_iteration=32,
    )
    steps = dqn_zoo_atari_total_train_env_steps(runtime_config)
    agent = DQNAtariAgent(config, runtime_config)
    env = make_atari_adapter(AleAtariConfig(game="Pong", frame_skip=config.NUM_ACTION_REPEATS))

    final_state, metrics = jax.jit(lambda key: run(agent, env, key, steps=steps, gamma=GAMMA))(jax.random.key(0))

    for key_name in ("loss", "epsilon", "loop/reward", "loop/episode_end", "loop/episode_return"):
        assert metrics[key_name].shape == (steps,), key_name
    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert jnp.any(metrics["loss"] != 0.0), "the replay warmup completed, so the learn path must have fired"
    assert int(final_state.agent_state.buffer_state.count) == steps - 1
