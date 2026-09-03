
from typing import cast

import jax
from rl_agents.dqn_atari import DQNAtariConfig, dqn_atari_runtime_from_dqn_zoo, make_train
from rl_components.atari_ale import AleAtariConfig, make_atari_adapter
from rl_components.gym_env import DiscreteActionSpace, GymEnv
from rl_components.gymnax_bridge import make_gymnax_compat_env


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
        num_train_frames_per_iteration=16,
    )
    # The bridge reports a union action space because EnvSpec decides at runtime;
    # an ALE env is always discrete.
    env = cast(
        GymEnv[DiscreteActionSpace],
        make_gymnax_compat_env(make_atari_adapter(AleAtariConfig(game="Pong"))),
    )

    train = make_train(config, runtime_config, env=env, env_params=None)
    out = jax.jit(train)(jax.random.key(0))

    metrics = out["metrics"]
    assert "loss" in metrics
