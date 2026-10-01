"""Each agent config's traced fields match the dynamic set the batch trainer derives."""

import pytest
from research_runner.config import _traced_paths
from rl_agents.batch_trainer import _algorithm_metadata
from rl_agents.double_dqn import DoubleDQNConfig
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.dueling_dqn import DuelingDQNConfig
from rl_agents.greedy_ac import GACAgent, GACConfig, GACHypers
from rl_agents.ppo import PPOAgent, PPOConfig, PPOHypers
from rl_agents.qrc import QRCAgent, QRCConfig, QRCHypers
from rl_agents.replay_q_agent import ReplayQHypers
from rl_agents.sac import SACAgent, SACConfig, SACHypers
from rl_agents.sac_rc import SACRCAgent, SACRCConfig, SACRCHypers
from rl_agents.td3 import TD3Agent, TD3Config, TD3Hypers

# Fields the derivation marks dynamic but the config keeps static: PPO checks REWARD_SCALE in Python.
_STATIC_EXCEPTIONS = {"reward_scale", "target_entropy"}

# Double and Dueling DQN share DQNAgent's ReplayQHypers and declare no init-swept fields.
_CASES = [
    pytest.param(DQNConfig, DQNAgent, ReplayQHypers, id="dqn"),
    pytest.param(DoubleDQNConfig, DQNAgent, ReplayQHypers, id="double-dqn"),
    pytest.param(DuelingDQNConfig, DQNAgent, ReplayQHypers, id="dueling-dqn"),
    pytest.param(QRCConfig, QRCAgent, QRCHypers, id="qrc"),
    pytest.param(TD3Config, TD3Agent, TD3Hypers, id="td3"),
    pytest.param(GACConfig, GACAgent, GACHypers, id="gac"),
    pytest.param(PPOConfig, PPOAgent, PPOHypers, id="ppo"),
    pytest.param(SACConfig, SACAgent, SACHypers, id="sac"),
    pytest.param(SACRCConfig, SACRCAgent, SACRCHypers, id="sac-rc"),
]


@pytest.mark.parametrize(("config_cls", "agent_cls", "hypers_cls"), _CASES)
def test_traced_fields_equal_the_derived_dynamic_set(config_cls: type, agent_cls: type, hypers_cls: type) -> None:
    derived = _algorithm_metadata(agent_cls, config_cls, hypers_cls).dynamic_keys - {"gamma"} - _STATIC_EXCEPTIONS

    assert {path.lower() for path in _traced_paths(config_cls())} == derived


_REPLAY_Q = {
    "EPSILON_END", "EPSILON_FRACTION", "EPSILON_START", "LEARNING_RATE",
    "LEARNING_STARTS", "TARGET_NETWORK_FREQUENCY", "TAU", "TRAIN_FREQUENCY",
}
_EXPECTED_TRACED = [
    pytest.param(DQNConfig, _REPLAY_Q, id="dqn"),
    pytest.param(DoubleDQNConfig, _REPLAY_Q, id="double-dqn"),
    pytest.param(DuelingDQNConfig, _REPLAY_Q, id="dueling-dqn"),
    pytest.param(
        QRCConfig,
        {"BETA", "EPSILON_END", "EPSILON_FRACTION", "EPSILON_START", "LEARNING_RATE", "LEARNING_STARTS", "TRAIN_FREQUENCY"},
        id="qrc",
    ),
    pytest.param(
        TD3Config,
        {"EXPLORATION_NOISE", "LEARNING_RATE", "LEARNING_STARTS", "NOISE_CLIP", "POLICY_DELAY", "POLICY_NOISE", "TAU", "TRAIN_FREQUENCY"},
        id="td3",
    ),
    pytest.param(
        GACConfig,
        {
            "ACTOR_LEARNING_RATE", "ENTROPY_WEIGHT", "H_REGULARIZATION", "INFERENCE_SIGMA_MIN",
            "LEARNING_RATE", "LEARNING_STARTS", "TRAINING_SIGMA_MIN", "TRAIN_FREQUENCY",
        },
        id="gac",
    ),
    pytest.param(
        PPOConfig,
        {"CLIP_EPS", "ENT_COEF", "GAE_LAMBDA", "LEARNING_RATE", "MAX_GRAD_NORM", "OBS_NORM_CLIP", "OBS_NORM_EPS", "VF_COEF"},
        id="ppo",
    ),
    pytest.param(SACConfig, {"ALPHA", "LEARNING_RATE", "LEARNING_STARTS", "TAU", "TRAIN_FREQUENCY"}, id="sac"),
    pytest.param(SACRCConfig, {"ALPHA", "BETA", "LEARNING_RATE", "LEARNING_STARTS", "TRAIN_FREQUENCY"}, id="sac-rc"),
]


@pytest.mark.parametrize(("config_cls", "expected"), _EXPECTED_TRACED)
def test_each_config_traces_exactly_its_pinned_fields(config_cls: type, expected: set[str]) -> None:
    assert set(_traced_paths(config_cls())) == expected
