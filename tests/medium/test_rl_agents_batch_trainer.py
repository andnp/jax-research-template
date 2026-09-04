from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import gymnax
import jax
import pytest
from experiment_definition import ParameterValue
from research_runner import ExecutionContext, RunPoint
from rl_agents.batch_trainer import AlgorithmSpec, make_batch_trainer
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.greedy_ac import GACAgent, GACConfig, GACHypers
from rl_agents.ppo import PPOAgent, PPOHypers
from rl_agents.qrc import QRCAgent, QRCConfig, QRCHypers
from rl_agents.replay_q_agent import ReplayQHypers
from rl_agents.sac_rc import SACRCAgent, SACRCConfig, SACRCHypers
from rl_agents.td3 import TD3Agent, TD3Config, TD3Hypers
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.types import PPOConfig


def _make_env(config: Mapping[str, ParameterValue]):
    env_name = config["env_name"]
    assert isinstance(env_name, str)
    raw_env, env_params = gymnax.make(env_name)
    env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_gymnax_env(raw_env))
    if env.spec(env_params).num_actions is None:
        env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_action_normalization_wrapper(env))
    return env, env_params


DQN_SPEC = AlgorithmSpec("dqn", DQNAgent, DQNConfig, ReplayQHypers)
GAC_SPEC = AlgorithmSpec("gac", GACAgent, GACConfig, GACHypers)
PPO_SPEC = AlgorithmSpec("ppo", PPOAgent, PPOConfig, PPOHypers)
QRC_SPEC = AlgorithmSpec("qrc", QRCAgent, QRCConfig, QRCHypers)
SAC_RC_SPEC = AlgorithmSpec("sac-rc", SACRCAgent, SACRCConfig, SACRCHypers)
TD3_SPEC = AlgorithmSpec("td3", TD3Agent, TD3Config, TD3Hypers)

# The dynamic_keys each *_batch.py module hand-writes today, kept here only to prove the
# derivation in AlgorithmSpec.parameter_schema reproduces (or knowingly diverges from) it.
_HAND_WRITTEN_DYNAMIC_KEYS: Mapping[str, frozenset[str]] = {
    "dqn": frozenset(
        {
            "learning_rate",
            "gamma",
            "learning_starts",
            "train_frequency",
            "target_network_frequency",
            "tau",
            "epsilon_start",
            "epsilon_end",
            "epsilon_fraction",
        }
    ),
    "gac": frozenset(
        {
            "learning_rate",
            "actor_learning_rate",
            "gamma",
            "learning_starts",
            "train_frequency",
            "entropy_weight",
            "training_sigma_min",
            "inference_sigma_min",
            "h_regularization",
        }
    ),
    "ppo": frozenset(
        {
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_eps",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "obs_norm_eps",
            "obs_norm_clip",
        }
    ),
    "qrc": frozenset(
        {
            "learning_rate",
            "gamma",
            "learning_starts",
            "train_frequency",
            "epsilon_start",
            "epsilon_end",
            "epsilon_fraction",
            "beta",
        }
    ),
    "sac-rc": frozenset(
        {
            "learning_rate",
            "gamma",
            "learning_starts",
            "train_frequency",
            "alpha",
            "target_entropy",
            "beta",
        }
    ),
    "td3": frozenset(
        {
            "learning_rate",
            "learning_starts",
            "train_frequency",
            "tau",
            "policy_delay",
            "exploration_noise",
            "policy_noise",
            "noise_clip",
        }
    ),
}


class TestDynamicKeyDerivation:
    def test_td3_gains_only_gamma(self) -> None:
        """TD3's hand-written schema omits gamma; the derivation adds it and nothing else."""
        derived = TD3_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["td3"]
        assert derived == hand | {"gamma"}

    def test_dqn_matches_its_hand_written_schema(self) -> None:
        """DQN already hand-lists gamma, so the derivation reproduces its set exactly."""
        derived = DQN_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["dqn"]
        assert "gamma" in hand
        assert derived == hand

    def test_qrc_matches_its_hand_written_schema(self) -> None:
        """QRC already hand-lists gamma, so the derivation reproduces its set exactly."""
        derived = QRC_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["qrc"]
        assert "gamma" in hand
        assert derived == hand

    @pytest.mark.parametrize("spec", [DQN_SPEC, QRC_SPEC], ids=["dqn", "qrc"])
    def test_the_scan_length_is_never_batchable(self, spec: AlgorithmSpec) -> None:
        """``TOTAL_TIMESTEPS`` rides these hypers, but the loop needs it as a Python int.

        The epsilon schedule divides by it, which is why it sits on the hypers struct at
        all. It is also the ``lax.scan`` length handed to ``run``, and a traced value
        there fails to compile -- so declaring it batchable would promise a sweep the
        trainer cannot serve. The scan boundary wins.
        """
        assert "total_timesteps" not in spec.parameter_schema.dynamic_keys

    def test_ppo_gains_reward_scale(self) -> None:
        """PPOHypers carries REWARD_SCALE, which the hand-written schema omits."""
        derived = PPO_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["ppo"]
        assert derived == hand | {"reward_scale"}

    def test_gac_matches_its_hand_written_schema(self) -> None:
        """GACConfig.ACTOR_LR was renamed to ACTOR_LEARNING_RATE, so the upcase rule holds.

        The derivation names this key from the Hypers field, lowercased -- now
        "actor_learning_rate", matching the hand-written schema exactly instead of
        diverging into "actor_lr".
        """
        derived = GAC_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["gac"]
        assert "actor_learning_rate" in hand
        assert "gamma" in hand
        assert derived == hand

    def test_sac_rc_matches_its_hand_written_schema(self) -> None:
        """SACRCAgent.INIT_SWEPT_FIELDS restores ALPHA and TARGET_ENTROPY to the derivation.

        Neither field rides SACRCHypers: ALPHA is never traced by ``step`` (see that
        struct's own docstring), and TARGET_ENTROPY is resolved through a Python ``is
        None`` check in ``init``. But ``init`` runs inside the vmapped region exactly as
        ``step`` does, so a per-arm traced value works there too -- these are genuinely
        sweepable, just never through the hypers struct. ``INIT_SWEPT_FIELDS`` names that
        fact directly, so the derivation now reproduces the hand-written schema exactly.
        """
        derived = SAC_RC_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["sac-rc"]
        assert "gamma" in hand
        assert derived == hand

    def test_gac_never_sweeps_actor_percentile_or_uniform_weight(self) -> None:
        """The two array-sizing fields stay out of the derived schema.

        GACHypers already omits both (see its own docstring), so this also guards against
        a future GACHypers change silently making them sweepable.
        """
        assert "actor_percentile" not in GAC_SPEC.parameter_schema.dynamic_keys
        assert "uniform_weight" not in GAC_SPEC.parameter_schema.dynamic_keys


class TestGenericBatchTrainerExecution:
    def test_dqn_trainer_runs_and_reports_declared_metrics(self, tmp_path: Path) -> None:
        """The generic trainer trains DQN and reports exactly its declared metrics plus reward."""
        static_config = {
            "algorithm": "dqn",
            "env_name": "CartPole-v1",
            "total_timesteps": 8,
            "buffer_size": 16,
            "batch_size": 2,
            "learning_starts": 2,
            "gamma": 0.99,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": 1e-3, "seed": index}) for index in (0, 1)
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="dqn-generic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(DQN_SPEC, _make_env)
        result = trainer(context)
        replay = trainer(context)

        assert set(result.metric_curves) == {"reward", "loss", "epsilon", "q_max"}
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
        assert result.metric_curves == replay.metric_curves

    def test_td3_trainer_sweeps_gamma_dynamically(self, tmp_path: Path) -> None:
        """TD3 could not vmap gamma before; the generic trainer treats it as a dynamic axis."""
        static_config = {
            "algorithm": "td3",
            "env_name": "Pendulum-v1",
            "total_timesteps": 8,
            "buffer_size": 16,
            "batch_size": 2,
            "learning_starts": 0,
            "learning_rate": 1e-3,
        }
        points = tuple(
            RunPoint(index, {**static_config, "gamma": gamma, "seed": index})
            for index, gamma in enumerate((0.9, 0.99))
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="td3-generic-gamma-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(TD3_SPEC, _make_env)
        result = trainer(context)

        assert set(result.metric_curves) == {"reward", "critic_loss", "actor_loss"}
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)

    def test_gac_trainer_runs_and_reports_declared_metrics(self, tmp_path: Path) -> None:
        """The generic trainer trains GAC with its actor_learning_rate-named dynamic key."""
        static_config = {
            "algorithm": "gac",
            "env_name": "MountainCarContinuous-v0",
            "total_timesteps": 8,
            "buffer_size": 16,
            "batch_size": 2,
            "learning_starts": 0,
            "num_samples": 4,
            "num_rand_actions": 2,
            "hidden_size": 8,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": 1e-3, "actor_learning_rate": 1e-3, "seed": index})
            for index in (0, 1)
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="gac-generic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(GAC_SPEC, _make_env)
        result = trainer(context)

        assert set(result.metric_curves) == {
            "reward",
            "critic_loss",
            "actor_loss",
            "critic_grad_norm",
            "actor_grad_norm",
        }
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
