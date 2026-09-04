from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
import jax.numpy as jnp
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run
from rl_components.types import PPOConfig

from rl_agents.ppo import PPOAgent

PPO_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    dynamic_keys=frozenset(
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
)


def _held_episode_returns(metrics: Mapping[str, jax.Array]):
    def hold(previous: jax.Array, step: tuple[jax.Array, jax.Array]):
        episode_return, episode_end = step
        current = jnp.where(episode_end, episode_return, previous)
        return current, current

    _, returns = jax.lax.scan(
        hold,
        jnp.zeros((), dtype=jnp.float32),
        (metrics["loop/episode_return"], metrics["loop/episode_end"]),
    )
    return returns


def _config_from(batch: BoundBatch, dynamic: Mapping[str, jax.Array]):
    defaults = PPOConfig()
    return PPOConfig(
        LR=cast(float, dynamic.get("learning_rate", batch.static_or("learning_rate", float, defaults.LR))),
        GAE_LAMBDA=cast(float, dynamic.get("gae_lambda", batch.static_or("gae_lambda", float, defaults.GAE_LAMBDA))),
        NUM_STEPS=batch.static_or("num_steps", int, defaults.NUM_STEPS),
        TOTAL_TIMESTEPS=batch.static("total_timesteps", int),
        UPDATE_EPOCHS=batch.static_or("update_epochs", int, defaults.UPDATE_EPOCHS),
        NUM_MINIBATCHES=batch.static_or("num_minibatches", int, defaults.NUM_MINIBATCHES),
        CLIP_EPS=cast(float, dynamic.get("clip_eps", batch.static_or("clip_eps", float, defaults.CLIP_EPS))),
        ENT_COEF=cast(float, dynamic.get("ent_coef", batch.static_or("ent_coef", float, defaults.ENT_COEF))),
        VF_COEF=cast(float, dynamic.get("vf_coef", batch.static_or("vf_coef", float, defaults.VF_COEF))),
        MAX_GRAD_NORM=cast(
            float, dynamic.get("max_grad_norm", batch.static_or("max_grad_norm", float, defaults.MAX_GRAD_NORM))
        ),
        REWARD_SCALE=batch.static_or("reward_scale", float, defaults.REWARD_SCALE),
        NORMALIZE_OBSERVATIONS=batch.static_or(
            "normalize_observations", bool, defaults.NORMALIZE_OBSERVATIONS
        ),
        OBS_NORM_EPS=cast(float, dynamic.get("obs_norm_eps", batch.static_or("obs_norm_eps", float, defaults.OBS_NORM_EPS))),
        OBS_NORM_CLIP=cast(float, dynamic.get("obs_norm_clip", batch.static_or("obs_norm_clip", float, defaults.OBS_NORM_CLIP))),
        ENV_NAME=batch.static("env_name", str),
        SEED=0,
    )


class PPOBatchTrainer:
    parameter_schema = PPO_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
        *,
        extra_metric_names: tuple[str, ...] = (),
    ) -> None:
        self._env_factory = env_factory
        self._extra_metric_names = extra_metric_names

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static_config)
        dynamic_axes = {
            key: batch.dynamic(key, int if key == "learning_starts" else float)
            for key in self.parameter_schema.dynamic_keys
            if key in batch.dynamic_axes
        }

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = _config_from(batch, dynamic)
            agent = PPOAgent(config)
            final_state, metrics = run(
                agent,
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=cast(float, dynamic.get("gamma", batch.static_or("gamma", float, 0.99))),
                env_params=env_params,
            )
            typed_metrics = cast(Mapping[str, jax.Array], metrics)
            return final_state.agent_state.train_state.params, {
                "returned_episode_returns": _held_episode_returns(typed_metrics),
                **{name: typed_metrics[name] for name in self._extra_metric_names},
            }

        final_params, curves = run_batch(train_one, dynamic_axes, batch.seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "ppo",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in curves.items()},
            artifacts={"final_policy_params": final_params},
        )


def make_ppo_batch_trainer(
    env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
    *,
    extra_metric_names: tuple[str, ...] = (),
) -> BatchTrainer:
    return PPOBatchTrainer(env_factory, extra_metric_names=extra_metric_names)
