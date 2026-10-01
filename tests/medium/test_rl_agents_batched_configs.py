"""Every agent config runs through one batch, its traced fields varying across the batch's runs."""

from collections.abc import Callable

import numpy as np
import pytest
from experiment_definition.experiment import Experiment
from research_runner.batch import _run_batch
from rl_agents.double_dqn import DoubleDQNConfig
from rl_agents.dqn import DQNConfig
from rl_agents.dueling_dqn import DuelingDQNConfig
from rl_agents.experiment import AgentConfig, RLConfig, rl_train
from rl_agents.greedy_ac import GACConfig
from rl_agents.ppo import PPOConfig
from rl_agents.qrc import QRCConfig
from rl_agents.sac import SACConfig
from rl_agents.sac_rc import SACRCConfig
from rl_agents.td3 import TD3Config
from rl_envs.env_config import GymnaxConfig

_STEPS = 4
_REPLAY = {"TOTAL_TIMESTEPS": _STEPS, "BUFFER_SIZE": 8, "BATCH_SIZE": 2, "LEARNING_STARTS": 0}
_CARTPOLE = GymnaxConfig("CartPole-v1")
_PENDULUM = GymnaxConfig("Pendulum-v1", normalized=True)

_CASES = [
    pytest.param(DQNConfig, _REPLAY, _CARTPOLE, id="dqn"),
    pytest.param(DoubleDQNConfig, _REPLAY, _CARTPOLE, id="double-dqn"),
    pytest.param(DuelingDQNConfig, _REPLAY, _CARTPOLE, id="dueling-dqn"),
    pytest.param(QRCConfig, {**_REPLAY, "HIDDEN_WIDTH": 8}, _CARTPOLE, id="qrc"),
    pytest.param(
        GACConfig,
        {**_REPLAY, "NUM_SAMPLES": 2, "NUM_RAND_ACTIONS": 1, "HIDDEN_SIZE": 8},
        GymnaxConfig("MountainCarContinuous-v0", normalized=True),
        id="gac",
    ),
    pytest.param(
        PPOConfig, {"TOTAL_TIMESTEPS": _STEPS, "NUM_STEPS": 2, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1}, _CARTPOLE, id="ppo"
    ),
    pytest.param(TD3Config, _REPLAY, _PENDULUM, id="td3"),
    pytest.param(SACConfig, _REPLAY, _PENDULUM, id="sac"),
    pytest.param(SACRCConfig, _REPLAY, _PENDULUM, id="sac-rc"),
]


@pytest.mark.parametrize(("agent_cls", "fields", "env"), _CASES)
def test_runs_differing_in_traced_fields_share_one_batch(
    agent_cls: Callable[..., AgentConfig[object]], fields: dict[str, int], env: GymnaxConfig
) -> None:
    configs = [
        RLConfig(agent=agent_cls(**fields, LEARNING_RATE=1e-3), env=env, gamma=0.9),
        RLConfig(agent=agent_cls(**fields, LEARNING_RATE=3e-4), env=env, gamma=0.99),
    ]
    metrics = Experiment("agents").add_metric("loop/reward").metrics

    outputs = _run_batch(rl_train, metrics, configs, [0, 1])

    assert outputs["loop/reward"].shape == (2, _STEPS)
    assert np.isfinite(outputs["loop/reward"]).all()
