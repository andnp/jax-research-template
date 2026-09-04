from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run

from rl_agents.td3 import TD3Agent, TD3Config

TD3_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    static_keys=frozenset(
        {
            "algorithm",
            "env_name",
            "total_timesteps",
            "buffer_size",
            "batch_size",
            "gamma",
        }
    ),
    dynamic_keys=frozenset(
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
)


class TD3BatchTrainer:
    parameter_schema = TD3_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
    ) -> None:
        self._env_factory = env_factory

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static_config)
        dynamic_axes = {
            key: batch.dynamic(key, int if key in {"learning_starts", "train_frequency", "policy_delay"} else float)
            for key in self.parameter_schema.dynamic_keys
            if key in batch.dynamic_axes
        }

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = TD3Config(
                LR=cast(float, dynamic["learning_rate"]),
                BUFFER_SIZE=batch.static("buffer_size", int),
                BATCH_SIZE=batch.static("batch_size", int),
                TOTAL_TIMESTEPS=batch.static("total_timesteps", int),
                LEARNING_STARTS=cast(int, dynamic.get("learning_starts", 25_000)),
                TRAIN_FREQUENCY=cast(int, dynamic.get("train_frequency", 1)),
                TAU=cast(float, dynamic.get("tau", 0.005)),
                POLICY_DELAY=cast(int, dynamic.get("policy_delay", 2)),
                EXPLORATION_NOISE=cast(float, dynamic.get("exploration_noise", 0.1)),
                POLICY_NOISE=cast(float, dynamic.get("policy_noise", 0.2)),
                NOISE_CLIP=cast(float, dynamic.get("noise_clip", 0.5)),
                ENV_NAME=batch.static("env_name", str),
                SEED=0,
            )
            agent = TD3Agent(config)
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
                "algorithm": "td3",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )

def make_td3_batch_trainer(
    env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
) -> BatchTrainer:
    return TD3BatchTrainer(env_factory)
