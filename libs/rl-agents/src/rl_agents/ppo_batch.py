from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
import jax.numpy as jnp
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run
from rl_components.types import PPOConfig

from rl_agents.ppo import PPOAgent

PPO_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    static_keys=frozenset(
        {
            "algorithm",
            "env_name",
            "num_steps",
            "total_timesteps",
            "update_epochs",
            "num_minibatches",
            "gamma",
            "clip_eps",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "reward_scale",
            "normalize_observations",
            "obs_norm_eps",
            "obs_norm_clip",
        }
    ),
    dynamic_keys=frozenset({"learning_rate"}),
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


def _config_from(batch: BoundBatch, learning_rate: float | jax.Array):
    return PPOConfig(
        LR=cast(float, learning_rate),
        NUM_STEPS=batch.static("num_steps", int),
        TOTAL_TIMESTEPS=batch.static("total_timesteps", int),
        UPDATE_EPOCHS=batch.static("update_epochs", int),
        NUM_MINIBATCHES=batch.static("num_minibatches", int),
        CLIP_EPS=batch.static("clip_eps", float),
        ENT_COEF=batch.static("ent_coef", float),
        VF_COEF=batch.static("vf_coef", float),
        MAX_GRAD_NORM=batch.static("max_grad_norm", float),
        REWARD_SCALE=batch.static("reward_scale", float),
        NORMALIZE_OBSERVATIONS=batch.static("normalize_observations", bool),
        OBS_NORM_EPS=batch.static("obs_norm_eps", float),
        OBS_NORM_CLIP=batch.static("obs_norm_clip", float),
        ENV_NAME=batch.static("env_name", str),
        SEED=0,
    )


class PPOBatchTrainer:
    parameter_schema = PPO_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[str], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]],
    ) -> None:
        self._env_factory = env_factory

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static("env_name", str))
        dynamic_axes = {"learning_rate": batch.dynamic("learning_rate", float)}

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = _config_from(batch, dynamic["learning_rate"])
            agent = PPOAgent(config)
            final_state, metrics = run(
                agent,
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=batch.static("gamma", float),
                env_params=env_params,
            )
            return final_state.agent_state.train_state.params, _held_episode_returns(
                cast(Mapping[str, jax.Array], metrics)
            )

        final_params, curves = run_batch(train_one, dynamic_axes, batch.seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "ppo",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={"returned_episode_returns": curves.tolist()},
            artifacts={"final_policy_params": final_params},
        )


def make_ppo_batch_trainer(
    env_factory: Callable[[str], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]],
) -> BatchTrainer:
    return PPOBatchTrainer(env_factory)
