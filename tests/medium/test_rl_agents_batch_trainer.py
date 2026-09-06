from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import gymnax
import jax
import jax.numpy as jnp
import pytest
from experiment_definition import ParameterValue
from research_runner import BoundBatch, ExecutionContext, RunPoint
from rl_agents.batch_registry import ALGORITHM_TRAINERS, make_algorithm_batch_trainer
from rl_agents.batch_trainer import AlgorithmSpec, _config_kwargs, make_batch_trainer
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.greedy_ac import GACAgent, GACConfig, GACHypers
from rl_agents.ppo import PPOAgent, PPOHypers
from rl_agents.qrc import QRCAgent, QRCConfig, QRCHypers
from rl_agents.replay_q_agent import ReplayQHypers
from rl_agents.sac import SACAgent, SACConfig, SACHypers
from rl_agents.sac_rc import SACRCAgent, SACRCConfig, SACRCHypers
from rl_agents.td3 import TD3Agent, TD3Config, TD3Hypers
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol, EnvStep
from rl_components.types import PPOConfig
from rl_envs.gymnax_bridge import make_gymnax_env


def _make_env(config: Mapping[str, ParameterValue]):
    env_name = config["env_name"]
    assert isinstance(env_name, str)
    raw_env, env_params = gymnax.make(env_name)
    env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_gymnax_env(raw_env))
    if env.spec(env_params).num_actions is None:
        env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_action_normalization_wrapper(env))
    return env, env_params


class _DiagnosticEnv:
    """Wraps an env to forward a fixed ``env/`` diagnostic schema, as a real adapter would."""

    def __init__(self, env: EnvProtocol[jax.Array, object, jax.Array, object]) -> None:
        self._env = env

    def spec(self, params: object | None = None):
        return self._env.spec(params)

    def reset(self, key: jax.Array, params: object | None = None):
        return self._env.reset(key, params)

    def step(
        self,
        key: jax.Array,
        state: object,
        action: jax.Array,
        params: object | None = None,
    ):
        transition = self._env.step(key, state, action, params)
        return EnvStep(
            observation=transition.observation,
            state=transition.state,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            info={
                "missing_fraction": jnp.asarray(0.25, dtype=jnp.float32),
                "missing_count": jnp.asarray(1.0, dtype=jnp.float32),
            },
        )


def _make_diagnostic_env(config: Mapping[str, ParameterValue]):
    env, env_params = _make_env(config)
    return _DiagnosticEnv(env), env_params


DQN_SPEC = AlgorithmSpec("dqn", DQNAgent, DQNConfig, ReplayQHypers)
GAC_SPEC = AlgorithmSpec("gac", GACAgent, GACConfig, GACHypers)
PPO_SPEC = AlgorithmSpec("ppo", PPOAgent, PPOConfig, PPOHypers)
QRC_SPEC = AlgorithmSpec("qrc", QRCAgent, QRCConfig, QRCHypers)
SAC_SPEC = AlgorithmSpec("sac", SACAgent, SACConfig, SACHypers)
SAC_RC_SPEC = AlgorithmSpec("sac-rc", SACRCAgent, SACRCConfig, SACRCHypers)
TD3_SPEC = AlgorithmSpec("td3", TD3Agent, TD3Config, TD3Hypers)

# The dynamic_keys each now-deleted hand-written *_batch.py module used to declare, kept
# here only to prove the derivation in AlgorithmSpec.parameter_schema reproduces (or
# knowingly diverges from) it. Captured from the modules before their deletion.
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
    "sac": frozenset({"learning_rate", "learning_starts", "train_frequency", "tau"}),
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

    def test_sac_gains_gamma_alpha_and_target_entropy(self) -> None:
        """SACAgent.INIT_SWEPT_FIELDS restores ALPHA and TARGET_ENTROPY to the derivation.

        Mirrors SAC-RC exactly: neither field rides SACHypers, but ``init`` runs inside
        the vmapped region just as ``step`` does, so both are genuinely sweepable.
        """
        derived = SAC_SPEC.parameter_schema.dynamic_keys
        hand = _HAND_WRITTEN_DYNAMIC_KEYS["sac"]
        assert derived == hand | {"gamma", "alpha", "target_entropy"}

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


class TestConfigKwargsResolution:
    """The generic ``_config_kwargs`` precedence contract the deleted ``_config_from``
    helpers used to prove individually: dynamic overrides static overrides default.
    """

    def test_prefers_dynamic_over_static_over_default(self) -> None:
        batch = BoundBatch(
            static_config={"env_name": "CartPole-v1", "learning_rate": 2e-3, "learning_starts": 3},
            dynamic_axes={},
            seeds=(0,),
            points=(),
        )
        static_kwargs = _config_kwargs(DQNConfig, batch, {})
        dynamic_kwargs = _config_kwargs(DQNConfig, batch, {"learning_rate": jnp.asarray(4e-3)})

        assert static_kwargs["LEARNING_RATE"] == 2e-3
        assert static_kwargs["LEARNING_STARTS"] == 3
        assert static_kwargs["TAU"] == DQNConfig().TAU
        assert dynamic_kwargs["LEARNING_RATE"] == jnp.asarray(4e-3)

    def test_requires_environment_name(self) -> None:
        batch = BoundBatch(static_config={}, dynamic_axes={}, seeds=(0,), points=())
        with pytest.raises(ValueError, match="missing static parameter 'env_name'"):
            _config_kwargs(DQNConfig, batch, {})

    def test_optional_field_distinguishes_missing_explicit_none_and_value(self) -> None:
        """SAC's TARGET_ENTROPY is ``float | None``; each source must reach the agent right.

        Missing from static config resolves to the field's own ``None`` default. Present
        and explicitly ``None`` must stay ``None`` too, rather than fail ``static_or``'s
        type coercion (which does not accept ``None`` for a ``float`` field). Present with
        a number coerces normally, and a dynamic override wins over all of it.
        """
        missing = _config_kwargs(
            SACConfig,
            BoundBatch(static_config={"env_name": "Pendulum-v1"}, dynamic_axes={}, seeds=(0,), points=()),
            {},
        )
        explicit_none = _config_kwargs(
            SACConfig,
            BoundBatch(
                static_config={"env_name": "Pendulum-v1", "target_entropy": None},
                dynamic_axes={},
                seeds=(0,),
                points=(),
            ),
            {},
        )
        explicit_value = _config_kwargs(
            SACConfig,
            BoundBatch(
                static_config={"env_name": "Pendulum-v1", "target_entropy": -1.5},
                dynamic_axes={},
                seeds=(0,),
                points=(),
            ),
            {},
        )
        dynamic_override = _config_kwargs(
            SACConfig,
            BoundBatch(
                static_config={"env_name": "Pendulum-v1", "target_entropy": -1.5},
                dynamic_axes={},
                seeds=(0,),
                points=(),
            ),
            {"target_entropy": jnp.asarray(-0.5)},
        )

        assert missing["TARGET_ENTROPY"] is None
        assert explicit_none["TARGET_ENTROPY"] is None
        assert explicit_value["TARGET_ENTROPY"] == -1.5
        assert dynamic_override["TARGET_ENTROPY"] == jnp.asarray(-0.5)

    def test_rejects_invalid_literal_value(self) -> None:
        """QRC's H_REGULARIZATION is a closed set; an unrecognised value must not pass
        through silently -- it used to be caught by ``qrc_batch.py``'s own hand check.
        """
        batch = BoundBatch(
            static_config={"env_name": "CartPole-v1", "h_regularization": "not-a-real-choice"},
            dynamic_axes={},
            seeds=(0,),
            points=(),
        )
        with pytest.raises(ValueError, match="H_REGULARIZATION"):
            _config_kwargs(QRCConfig, batch, {})

    def test_accepts_valid_literal_value(self) -> None:
        batch = BoundBatch(
            static_config={"env_name": "CartPole-v1", "network_preset": "nature_cnn"},
            dynamic_axes={},
            seeds=(0,),
            points=(),
        )
        assert _config_kwargs(DQNConfig, batch, {})["NETWORK_PRESET"] == "nature_cnn"


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

    def test_qrc_trainer_runs_and_reports_declared_metrics(self, tmp_path: Path) -> None:
        static_config = {
            "algorithm": "qrc",
            "env_name": "CartPole-v1",
            "total_timesteps": 8,
            "buffer_size": 16,
            "batch_size": 2,
            "learning_starts": 0,
            "hidden_width": 8,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": 1e-3, "seed": index})
            for index in (0, 1)
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="qrc-generic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(QRC_SPEC, _make_env)
        result = trainer(context)
        replay = trainer(context)

        assert set(result.metric_curves) == {"reward", "loss", "epsilon", "q_max"}
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
        assert result.metric_curves == replay.metric_curves

    def test_sac_rc_trainer_runs_and_reports_declared_metrics(self, tmp_path: Path) -> None:
        static_config = {
            "algorithm": "sac-rc",
            "env_name": "Pendulum-v1",
            "total_timesteps": 8,
            "buffer_size": 16,
            "batch_size": 2,
            "learning_starts": 0,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": rate, "seed": index})
            for index, rate in enumerate((1e-3, 1e-4))
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="sac-rc-generic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(SAC_RC_SPEC, _make_env)
        result = trainer(context)
        replay = trainer(context)

        assert set(result.metric_curves) == {"reward", "critic_loss", "actor_loss", "alpha_loss", "alpha"}
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
        assert result.metric_curves == replay.metric_curves

    def test_sac_trainer_runs_and_reports_declared_metrics(self, tmp_path: Path) -> None:
        static_config = {
            "algorithm": "sac",
            "env_name": "Pendulum-v1",
            "total_timesteps": 8,
            "buffer_size": 16,
            "batch_size": 2,
            "learning_starts": 2,
            "gamma": 0.99,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": rate, "seed": index})
            for index, rate in enumerate((1e-3, 1e-4))
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="sac-generic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(SAC_SPEC, _make_env)
        result = trainer(context)
        replay = trainer(context)

        assert set(result.metric_curves) == {"reward", "critic_loss", "actor_loss", "alpha_loss", "alpha"}
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
        assert result.metric_curves == replay.metric_curves

    def test_ppo_trainer_reports_returned_episode_returns_and_final_params(self, tmp_path: Path) -> None:
        """PPO's report shape genuinely differs: no "reward", an episode-return hold
        instead, and its final policy parameters as an artifact.
        """
        static_config = {
            "algorithm": "ppo",
            "env_name": "CartPole-v1",
            "num_steps": 4,
            "total_timesteps": 8,
            "update_epochs": 1,
            "num_minibatches": 1,
            "gamma": 0.99,
            "reward_scale": 1.0,
            "normalize_observations": False,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": rate, "seed": index})
            for index, rate in enumerate((1e-1, 1e-4))
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="ppo-generic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        result = make_batch_trainer(PPO_SPEC, _make_env)(context)

        assert set(result.metric_curves) == {
            "returned_episode_returns",
            "total_loss",
            "value_loss",
            "policy_loss",
            "entropy",
        }
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
        assert "final_policy_params" in result.artifacts

    def test_ppo_trainer_forwards_selected_environment_metrics(self, tmp_path: Path) -> None:
        """Forward selected environment diagnostics without changing PPO's batching."""
        static_config = {
            "algorithm": "ppo",
            "env_name": "CartPole-v1",
            "num_steps": 4,
            "total_timesteps": 8,
            "update_epochs": 1,
            "num_minibatches": 1,
            "gamma": 0.99,
            "reward_scale": 1.0,
            "normalize_observations": False,
        }
        points = tuple(
            RunPoint(index, {**static_config, "learning_rate": 1e-4, "seed": index})
            for index in (0, 1)
        )
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="ppo-diagnostic-test",
            static_config=static_config,
            points=points,
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        trainer = make_batch_trainer(
            PPO_SPEC,
            _make_diagnostic_env,
            extra_metric_names=("env/missing_fraction", "env/missing_count"),
        )
        result = trainer(context)
        replay = trainer(context)

        assert {"env/missing_fraction", "env/missing_count"} <= set(result.metric_curves)
        assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
        assert result.metric_curves == replay.metric_curves


class TestAlgorithmRegistry:
    """The behaviour ``batch_registry.py`` promises: one dispatcher, seven specs."""

    def test_all_seven_algorithms_are_registered(self) -> None:
        assert set(ALGORITHM_TRAINERS) == {"dqn", "gac", "ppo", "qrc", "sac", "sac-rc", "td3"}

    def test_rejects_invalid_configuration_at_construction(self) -> None:
        with pytest.raises(ValueError, match="at least one algorithm"):
            make_algorithm_batch_trainer(_make_env, ())
        with pytest.raises(ValueError, match="no batch trainer registered"):
            make_algorithm_batch_trainer(_make_env, ("unknown",))

    def test_rejects_invalid_runtime_algorithm(self, tmp_path: Path) -> None:
        trainer = make_algorithm_batch_trainer(_make_env, ("sac",))
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="invalid-algorithm-test",
            static_config={},
            points=(),
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )
        with pytest.raises(ValueError, match="'algorithm' must be text"):
            trainer(context)

    def test_extra_specs_registers_a_project_local_algorithm(self, tmp_path: Path) -> None:
        """A project's own AlgorithmSpec dispatches without touching ALGORITHM_TRAINERS."""
        local_spec = AlgorithmSpec("qrc-variant", QRCAgent, QRCConfig, QRCHypers)
        trainer = make_algorithm_batch_trainer(_make_env, ("qrc-variant",), extra_specs={"qrc-variant": local_spec})
        static_config = {
            "algorithm": "qrc-variant",
            "env_name": "CartPole-v1",
            "total_timesteps": 4,
            "buffer_size": 8,
            "batch_size": 2,
            "learning_starts": 0,
            "hidden_width": 8,
        }
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="extra-specs-test",
            static_config=static_config,
            points=(RunPoint(0, {**static_config, "learning_rate": 1e-3, "seed": 0}),),
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        result = trainer(context)

        assert set(result.metric_curves) == {"reward", "loss", "epsilon", "q_max"}

    def test_extra_specs_reports_returned_episode_returns_when_asked(self, tmp_path: Path) -> None:
        """The flag added for GenericBatchTrainer must reach callers of the dispatcher too."""
        trainer = make_algorithm_batch_trainer(_make_env, ("sac",), report_returned_episode_returns=True)
        static_config = {
            "algorithm": "sac",
            "env_name": "Pendulum-v1",
            "total_timesteps": 4,
            "buffer_size": 8,
            "batch_size": 2,
            "learning_starts": 0,
            "gamma": 0.99,
        }
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="returned-episode-returns-dispatch-test",
            static_config=static_config,
            points=(RunPoint(0, {**static_config, "learning_rate": 1e-3, "seed": 0}),),
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )

        result = trainer(context)

        assert "returned_episode_returns" in result.metric_curves

    def test_rejects_mixed_h_regularization_axis(self, tmp_path: Path) -> None:
        """The intersected schema must reject a varying h_regularization axis before
        dispatch can pass a numeric GAC weight to the categorical QRC contract.
        """
        static_config = {"algorithm": "qrc", "env_name": "CartPole-v1"}
        context = ExecutionContext(
            execution_id=1,
            experiment_id=1,
            experiment_name="mixed-regularization-test",
            static_config=static_config,
            points=(
                RunPoint(0, {**static_config, "h_regularization": "adamw", "seed": 0}),
                RunPoint(1, {**static_config, "h_regularization": "l2", "seed": 1}),
            ),
            execution_root=tmp_path,
            metrics_db_path=tmp_path / "metrics.sqlite",
        )
        trainer = make_algorithm_batch_trainer(_make_env, ("gac", "qrc"))

        with pytest.raises(ValueError, match="h_regularization"):
            trainer(context)

    def test_dispatches_every_registered_algorithm(self, tmp_path: Path) -> None:
        algorithms = ("dqn", "gac", "ppo", "qrc", "sac", "sac-rc", "td3")
        trainer = make_algorithm_batch_trainer(_make_env, algorithms)
        configs: Mapping[str, Mapping[str, ParameterValue]] = {
            "dqn": {
                "algorithm": "dqn",
                "env_name": "CartPole-v1",
                "total_timesteps": 4,
                "buffer_size": 8,
                "batch_size": 2,
                "learning_starts": 0,
                "learning_rate": 1e-3,
            },
            "gac": {
                "algorithm": "gac",
                "env_name": "MountainCarContinuous-v0",
                "total_timesteps": 4,
                "buffer_size": 8,
                "batch_size": 2,
                "learning_starts": 0,
                "num_samples": 2,
                "num_rand_actions": 1,
                "hidden_size": 8,
                "learning_rate": 1e-3,
            },
            "ppo": {
                "algorithm": "ppo",
                "env_name": "CartPole-v1",
                "num_steps": 2,
                "total_timesteps": 4,
                "update_epochs": 1,
                "num_minibatches": 1,
                "learning_rate": 1e-3,
            },
            "qrc": {
                "algorithm": "qrc",
                "env_name": "CartPole-v1",
                "total_timesteps": 4,
                "buffer_size": 8,
                "batch_size": 2,
                "learning_starts": 0,
                "hidden_width": 8,
                "learning_rate": 1e-3,
            },
            "sac": {
                "algorithm": "sac",
                "env_name": "Pendulum-v1",
                "total_timesteps": 4,
                "buffer_size": 8,
                "batch_size": 2,
                "learning_starts": 0,
                "learning_rate": 1e-3,
            },
            "sac-rc": {
                "algorithm": "sac-rc",
                "env_name": "Pendulum-v1",
                "total_timesteps": 4,
                "buffer_size": 8,
                "batch_size": 2,
                "learning_starts": 0,
                "learning_rate": 1e-3,
            },
            "td3": {
                "algorithm": "td3",
                "env_name": "Pendulum-v1",
                "total_timesteps": 4,
                "buffer_size": 8,
                "batch_size": 2,
                "learning_starts": 0,
                "learning_rate": 1e-3,
            },
        }
        metric_names = {
            "dqn": {"reward", "loss", "epsilon", "q_max"},
            "gac": {"reward", "critic_loss", "actor_loss", "critic_grad_norm", "actor_grad_norm"},
            "ppo": {"returned_episode_returns", "total_loss", "value_loss", "policy_loss", "entropy"},
            "qrc": {"reward", "loss", "epsilon", "q_max"},
            "sac": {"reward", "critic_loss", "actor_loss", "alpha_loss", "alpha"},
            "sac-rc": {"reward", "critic_loss", "actor_loss", "alpha_loss", "alpha"},
            "td3": {"reward", "critic_loss", "actor_loss"},
        }

        for algorithm in algorithms:
            static_config = configs[algorithm]
            context = ExecutionContext(
                execution_id=1,
                experiment_id=1,
                experiment_name=f"{algorithm}-registry-test",
                static_config=static_config,
                points=(RunPoint(0, {**static_config, "seed": 0}),),
                execution_root=tmp_path,
                metrics_db_path=tmp_path / f"{algorithm}.sqlite",
            )
            result = trainer(context)

            assert result.metadata["algorithm"] == algorithm
            assert set(result.metric_curves) == metric_names[algorithm]
            assert all(len(curve) == 4 for metric in result.metric_curves.values() for curve in metric)
