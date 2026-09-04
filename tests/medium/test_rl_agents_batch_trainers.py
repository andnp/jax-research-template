from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import gymnax
import jax
import jax.numpy as jnp
from experiment_definition import ParameterValue
from research_runner import ExecutionContext, RunPoint
from rl_agents.dqn_batch import DQNBatchTrainer
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
