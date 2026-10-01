"""Each agent config traces exactly the fields pinned here."""

import pytest
from research_runner.config import _traced_paths
from rl_agents.double_dqn import DoubleDQNConfig
from rl_agents.dqn import DQNConfig
from rl_agents.dueling_dqn import DuelingDQNConfig
from rl_agents.greedy_ac import GACConfig
from rl_agents.ppo import PPOConfig
from rl_agents.qrc import QRCConfig
from rl_agents.sac import SACConfig
from rl_agents.sac_rc import SACRCConfig
from rl_agents.td3 import TD3Config

_REPLAY_Q = {
    "epsilon_end", "epsilon_fraction", "epsilon_start", "learning_rate",
    "learning_starts", "target_network_frequency", "tau", "train_frequency",
}
_EXPECTED_TRACED = [
    pytest.param(DQNConfig, _REPLAY_Q, id="dqn"),
    pytest.param(DoubleDQNConfig, _REPLAY_Q, id="double-dqn"),
    pytest.param(DuelingDQNConfig, _REPLAY_Q, id="dueling-dqn"),
    pytest.param(
        QRCConfig,
        {"beta", "epsilon_end", "epsilon_fraction", "epsilon_start", "learning_rate", "learning_starts", "train_frequency"},
        id="qrc",
    ),
    pytest.param(
        TD3Config,
        {"EXPLORATION_NOISE", "learning_rate", "learning_starts", "NOISE_CLIP", "POLICY_DELAY", "POLICY_NOISE", "TAU", "TRAIN_FREQUENCY"},
        id="td3",
    ),
    pytest.param(
        GACConfig,
        {
            "ACTOR_LEARNING_RATE", "ENTROPY_WEIGHT", "H_REGULARIZATION", "INFERENCE_SIGMA_MIN",
            "learning_rate", "learning_starts", "TRAINING_SIGMA_MIN", "TRAIN_FREQUENCY",
        },
        id="gac",
    ),
    pytest.param(
        PPOConfig,
        {"CLIP_EPS", "ENT_COEF", "GAE_LAMBDA", "learning_rate", "MAX_GRAD_NORM", "OBS_NORM_CLIP", "OBS_NORM_EPS", "VF_COEF"},
        id="ppo",
    ),
    pytest.param(SACConfig, {"alpha", "learning_rate", "learning_starts", "tau", "train_frequency"}, id="sac"),
    pytest.param(SACRCConfig, {"ALPHA", "BETA", "learning_rate", "learning_starts", "TRAIN_FREQUENCY"}, id="sac-rc"),
]


@pytest.mark.parametrize(("config_cls", "expected"), _EXPECTED_TRACED)
def test_each_config_traces_exactly_its_pinned_fields(config_cls: type, expected: set[str]) -> None:
    assert set(_traced_paths(config_cls())) == expected
