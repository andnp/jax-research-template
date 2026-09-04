from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import gymnax
import jax
import jax.numpy as jnp
from experiment_definition import ParameterValue
from research_runner import ExecutionContext, RunPoint
from rl_agents.ppo_batch import PPOBatchTrainer
from rl_agents.sac_batch import SACBatchTrainer
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_components.gymnax_bridge import make_gymnax_env


def _make_env(config: Mapping[str, ParameterValue]):
    env_name = config["env_name"]
    assert isinstance(env_name, str)
    raw_env, env_params = gymnax.make(env_name)
    env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_gymnax_env(raw_env))
    if env.spec(env_params).num_actions is None:
        env = cast(EnvProtocol[jax.Array, object, jax.Array, object], make_action_normalization_wrapper(env))
    return env, env_params


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
