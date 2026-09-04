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
from rl_agents.dqn import DQNConfig
from rl_agents.dqn_batch import DQNBatchTrainer, _config_from
from rl_agents.gac_batch import GACBatchTrainer
from rl_agents.ppo_batch import PPOBatchTrainer
from rl_agents.sac_batch import SACBatchTrainer
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol, EnvStep
from rl_components.gymnax_bridge import make_gymnax_env


def _make_env(config: Mapping[str, ParameterValue]):
    env_name = config["env_name"]
    assert isinstance(env_name, str)
    raw_env, env_params = gymnax.make(env_name)
    env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_gymnax_env(raw_env))
    if env.spec(env_params).num_actions is None:
        env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_action_normalization_wrapper(env))
    return env, env_params


class _DiagnosticEnv:
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


def test_sac_batch_trainer_returns_vmapped_curves(tmp_path: Path) -> None:
    """
    Verify the core SAC trainer handles dynamic rates through the same port.
    """
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
        experiment_name="sac-batch-test",
        static_config=static_config,
        points=points,
        execution_root=tmp_path,
        metrics_db_path=tmp_path / "metrics.sqlite",
    )

    trainer = SACBatchTrainer(_make_env)
    result = trainer(context)
    replay = trainer(context)

    assert set(result.metric_curves) == {
        "reward",
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
    }
    assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
    assert all(jnp.isfinite(jnp.asarray(value)).all() for curve in result.metric_curves.values() for value in curve)
    assert result.metric_curves == replay.metric_curves


def test_dqn_batch_trainer_preserves_metrics_and_seed_batching(tmp_path: Path) -> None:
    """
    Verify DQN uses the shared batch boundary with stable metrics and seeds.

    Distinct seeds must remain independent while repeating the same execution is
    deterministic.
    """
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
        RunPoint(index, {**static_config, "learning_rate": 1e-3, "seed": index})
        for index in (0, 1)
    )
    context = ExecutionContext(
        execution_id=1,
        experiment_id=1,
        experiment_name="dqn-batch-test",
        static_config=static_config,
        points=points,
        execution_root=tmp_path,
        metrics_db_path=tmp_path / "metrics.sqlite",
    )

    trainer = DQNBatchTrainer(_make_env)
    result = trainer(context)
    replay = trainer(context)

    assert set(result.metric_curves) == {"reward", "loss", "epsilon", "q_max"}
    assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
    assert all(
        jnp.isfinite(jnp.asarray(value)).all()
        for metric in result.metric_curves.values()
        for value in metric
    )
    assert result.metric_curves == replay.metric_curves
    assert any(value != 0.0 for value in result.metric_curves["loss"][0])
    assert any(
        result.metric_curves[metric][0] != result.metric_curves[metric][1]
        for metric in result.metric_curves
    )


def test_dqn_batch_config_prefers_dynamic_static_and_default_values() -> None:
    """
    Verify DQN batch configuration precedence and static network selection.

    Dynamic sweep values override static values, which override agent defaults.
    """
    batch = BoundBatch(
        static_config={
            "env_name": "CartPole-v1",
            "buffer_size": 16,
            "batch_size": 2,
            "total_timesteps": 8,
            "learning_rate": 2e-3,
            "learning_starts": 3,
            "network_preset": "mlp",
        },
        dynamic_axes={},
        seeds=(0,),
        points=(),
    )

    static_config = _config_from(batch, {})
    dynamic_config = _config_from(batch, {"learning_rate": jnp.asarray(4e-3), "learning_starts": jnp.asarray(1)})

    assert static_config.LR == 2e-3
    assert static_config.LEARNING_STARTS == 3
    assert static_config.TAU == DQNConfig().TAU
    assert static_config.NETWORK_PRESET == "mlp"
    assert dynamic_config.LR == 4e-3
    assert dynamic_config.LEARNING_STARTS == 1


def test_dqn_batch_config_requires_environment_name() -> None:
    """
    Verify the DQN configuration and result metadata share one required environment key.
    """
    batch = BoundBatch(static_config={}, dynamic_axes={}, seeds=(0,), points=())

    with pytest.raises(ValueError, match="missing static parameter 'env_name'"):
        _config_from(batch, {})


def test_gac_batch_trainer_preserves_metrics_and_seed_batching(tmp_path: Path) -> None:
    """
    Verify GAC uses the shared batch boundary with its complete metric contract.

    Repeated seeds must replay exactly while distinct seeds remain independent.
    """
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
        RunPoint(index, {**static_config, "learning_rate": 1e-3, "seed": index})
        for index in (0, 1)
    )
    context = ExecutionContext(
        execution_id=1,
        experiment_id=1,
        experiment_name="gac-batch-test",
        static_config=static_config,
        points=points,
        execution_root=tmp_path,
        metrics_db_path=tmp_path / "metrics.sqlite",
    )

    trainer = GACBatchTrainer(_make_env)
    result = trainer(context)
    replay = trainer(context)

    assert set(result.metric_curves) == {
        "reward",
        "critic_loss",
        "actor_loss",
        "critic_grad_norm",
        "actor_grad_norm",
    }
    assert all(len(curve) == 8 for metric in result.metric_curves.values() for curve in metric)
    assert all(
        jnp.isfinite(jnp.asarray(sample)).all()
        for metric in result.metric_curves.values()
        for curve in metric
        for sample in curve
    )
    assert result.metric_curves == replay.metric_curves
    assert any(
        result.metric_curves[metric][0] != result.metric_curves[metric][1]
        for metric in result.metric_curves
    )


def test_ppo_batch_trainer_returns_vmapped_curves_and_artifacts(tmp_path: Path) -> None:
    """
    Verify the core PPO trainer handles dynamic rates without project vmap code.
    """
    static_config = {
        "algorithm": "ppo",
        "env_name": "CartPole-v1",
        "num_steps": 4,
        "total_timesteps": 8,
        "update_epochs": 1,
        "num_minibatches": 1,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "reward_scale": 1.0,
        "normalize_observations": False,
        "obs_norm_eps": 1e-8,
        "obs_norm_clip": 10.0,
    }
    points = tuple(
        RunPoint(index, {**static_config, "learning_rate": rate, "seed": index})
        for index, rate in enumerate((1e-1, 1e-4))
    )
    context = ExecutionContext(
        execution_id=1,
        experiment_id=1,
        experiment_name="ppo-batch-test",
        static_config=static_config,
        points=points,
        execution_root=tmp_path,
        metrics_db_path=tmp_path / "metrics.sqlite",
    )

    result = PPOBatchTrainer(_make_env)(context)

    curves = result.metric_curves["returned_episode_returns"]
    assert len(curves) == 2
    assert all(len(curve) == 8 for curve in curves)
    assert "final_policy_params" in result.artifacts


def test_ppo_batch_trainer_forwards_selected_environment_metrics(tmp_path: Path) -> None:
    """Forward selected environment diagnostics without changing PPO batching."""
    static_config = {
        "algorithm": "ppo",
        "env_name": "CartPole-v1",
        "num_steps": 4,
        "total_timesteps": 8,
        "update_epochs": 1,
        "num_minibatches": 1,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "reward_scale": 1.0,
        "normalize_observations": False,
        "obs_norm_eps": 1e-8,
        "obs_norm_clip": 10.0,
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

    trainer = PPOBatchTrainer(
        _make_diagnostic_env,
        extra_metric_names=("env/missing_fraction", "env/missing_count"),
    )
    result = trainer(context)
    replay = trainer(context)

    assert set(result.metric_curves) == {
        "returned_episode_returns",
        "env/missing_fraction",
        "env/missing_count",
    }
    assert all(
        len(curve) == 8 for metric in result.metric_curves.values() for curve in metric
    )
    assert all(
        jnp.isfinite(jnp.asarray(value)).all()
        for metric in result.metric_curves.values()
        for curve in metric
        for value in curve
    )
    assert result.metric_curves == replay.metric_curves
    assert "final_policy_params" in result.artifacts
