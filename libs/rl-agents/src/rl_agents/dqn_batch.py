from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run

from rl_agents.dqn import DQNAgent, DQNConfig

DQN_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    dynamic_keys=frozenset(
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
)


def _config_from(batch: BoundBatch, dynamic: Mapping[str, jax.Array]):
    return DQNConfig(
        LR=cast(float, dynamic["learning_rate"]),
        BUFFER_SIZE=batch.static("buffer_size", int),
        BATCH_SIZE=batch.static("batch_size", int),
        TOTAL_TIMESTEPS=batch.static("total_timesteps", int),
        LEARNING_STARTS=cast(int, dynamic.get("learning_starts", 1_000)),
        TRAIN_FREQUENCY=cast(int, dynamic.get("train_frequency", 1)),
        TARGET_NETWORK_FREQUENCY=cast(int, dynamic.get("target_network_frequency", 1_000)),
        TAU=cast(float, dynamic.get("tau", 1.0)),
        EPSILON_START=cast(float, dynamic.get("epsilon_start", 1.0)),
        EPSILON_END=cast(float, dynamic.get("epsilon_end", 0.05)),
        EPSILON_FRACTION=cast(float, dynamic.get("epsilon_fraction", 0.5)),
        ENV_NAME=batch.static("env_name", str),
        SEED=0,
        NETWORK_PRESET="mlp",
    )


class DQNBatchTrainer:
    parameter_schema = DQN_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
    ) -> None:
        self._env_factory = env_factory

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static_config)
        integer_keys = {"learning_starts", "train_frequency", "target_network_frequency"}
        dynamic_axes = {
            key: batch.dynamic(key, int if key in integer_keys else float)
            for key in self.parameter_schema.dynamic_keys
            if key in batch.dynamic_axes
        }

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = _config_from(batch, dynamic)
            _, metrics = run(
                DQNAgent(config),
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=cast(float, dynamic.get("gamma", 0.99)),
                env_params=env_params,
            )
            return {
                "reward": metrics["loop/reward"],
                "loss": metrics["loss"],
                "epsilon": metrics["epsilon"],
                "q_max": metrics["q_max"],
            }

        metrics = run_batch(train_one, dynamic_axes, batch.seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "dqn",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )


def make_dqn_batch_trainer(
    env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
) -> BatchTrainer:
    return DQNBatchTrainer(env_factory)
