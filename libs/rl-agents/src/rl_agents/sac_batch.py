from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run

from rl_agents.sac import SACAgent, SACConfig

SAC_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    dynamic_keys=frozenset({"learning_rate", "learning_starts", "train_frequency", "tau"}),
)


class SACBatchTrainer:
    parameter_schema = SAC_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
    ) -> None:
        self._env_factory = env_factory

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static_config)
        dynamic_axes = {
            key: batch.dynamic(key, int if key in {"learning_starts", "train_frequency"} else float)
            for key in self.parameter_schema.dynamic_keys
            if key in batch.dynamic_axes
        }

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = SACConfig(
                LR=cast(float, dynamic["learning_rate"]),
                BUFFER_SIZE=batch.static_or("buffer_size", int, 100_000),
                BATCH_SIZE=batch.static_or("batch_size", int, 256),
                TOTAL_TIMESTEPS=batch.static("total_timesteps", int),
                LEARNING_STARTS=cast(int, dynamic.get("learning_starts", 5_000)),
                TRAIN_FREQUENCY=cast(int, dynamic.get("train_frequency", 1)),
                TAU=cast(float, dynamic.get("tau", 0.005)),
                ALPHA=batch.static_or("alpha", float, 0.2),
                ENV_NAME=batch.static("env_name", str),
                SEED=0,
            )
            agent = SACAgent(config)
            _, metrics = run(
                agent,
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=batch.static("gamma", float),
                env_params=env_params,
            )
            return {
                "reward": metrics["loop/reward"],
                "critic_loss": metrics["critic_loss"],
                "actor_loss": metrics["actor_loss"],
            }

        metrics = run_batch(train_one, dynamic_axes, batch.seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "sac",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )


def make_sac_batch_trainer(
    env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
) -> BatchTrainer:
    return SACBatchTrainer(env_factory)
