from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

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
    defaults = DQNConfig()
    network_preset = batch.static_or("network_preset", str, defaults.NETWORK_PRESET)
    if network_preset not in ("mlp", "nature_cnn"):
        raise ValueError(f"unsupported DQN network preset: {network_preset!r}")

    return DQNConfig(
        LR=cast(float, dynamic.get("learning_rate", batch.static_or("learning_rate", float, defaults.LR))),
        BUFFER_SIZE=batch.static_or("buffer_size", int, defaults.BUFFER_SIZE),
        BATCH_SIZE=batch.static_or("batch_size", int, defaults.BATCH_SIZE),
        TOTAL_TIMESTEPS=batch.static_or("total_timesteps", int, defaults.TOTAL_TIMESTEPS),
        LEARNING_STARTS=cast(
            int, dynamic.get("learning_starts", batch.static_or("learning_starts", int, defaults.LEARNING_STARTS))
        ),
        TRAIN_FREQUENCY=cast(
            int, dynamic.get("train_frequency", batch.static_or("train_frequency", int, defaults.TRAIN_FREQUENCY))
        ),
        TARGET_NETWORK_FREQUENCY=cast(
            int,
            dynamic.get(
                "target_network_frequency",
                batch.static_or("target_network_frequency", int, defaults.TARGET_NETWORK_FREQUENCY),
            ),
        ),
        TAU=cast(float, dynamic.get("tau", batch.static_or("tau", float, defaults.TAU))),
        EPSILON_START=cast(
            float, dynamic.get("epsilon_start", batch.static_or("epsilon_start", float, defaults.EPSILON_START))
        ),
        EPSILON_END=cast(float, dynamic.get("epsilon_end", batch.static_or("epsilon_end", float, defaults.EPSILON_END))),
        EPSILON_FRACTION=cast(
            float, dynamic.get("epsilon_fraction", batch.static_or("epsilon_fraction", float, defaults.EPSILON_FRACTION))
        ),
        ENV_NAME=batch.static_or("env_name", str, defaults.ENV_NAME),
        SEED=batch.static_or("seed", int, defaults.SEED),
        NETWORK_PRESET=cast(Literal["mlp", "nature_cnn"], network_preset),
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
                gamma=cast(float, dynamic.get("gamma", batch.static_or("gamma", float, 0.99))),
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
