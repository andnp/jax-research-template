from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run

from rl_agents.qrc import QRCAgent, QRCConfig

QRC_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    dynamic_keys=frozenset(
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
)


def _config_from(batch: BoundBatch, dynamic: Mapping[str, jax.Array]):
    defaults = QRCConfig()
    h_regularization = batch.static_or("h_regularization", str, defaults.H_REGULARIZATION)
    if h_regularization not in ("adamw", "l2"):
        raise ValueError(f"unsupported QRC h regularization: {h_regularization!r}")

    return QRCConfig(
        LEARNING_RATE=cast(float, dynamic.get("learning_rate", batch.static_or("learning_rate", float, defaults.LEARNING_RATE))),
        BUFFER_SIZE=batch.static_or("buffer_size", int, defaults.BUFFER_SIZE),
        BATCH_SIZE=batch.static_or("batch_size", int, defaults.BATCH_SIZE),
        TOTAL_TIMESTEPS=batch.static_or("total_timesteps", int, defaults.TOTAL_TIMESTEPS),
        LEARNING_STARTS=cast(
            int, dynamic.get("learning_starts", batch.static_or("learning_starts", int, defaults.LEARNING_STARTS))
        ),
        TRAIN_FREQUENCY=cast(
            int, dynamic.get("train_frequency", batch.static_or("train_frequency", int, defaults.TRAIN_FREQUENCY))
        ),
        EPSILON_START=cast(
            float, dynamic.get("epsilon_start", batch.static_or("epsilon_start", float, defaults.EPSILON_START))
        ),
        EPSILON_END=cast(float, dynamic.get("epsilon_end", batch.static_or("epsilon_end", float, defaults.EPSILON_END))),
        EPSILON_FRACTION=cast(
            float, dynamic.get("epsilon_fraction", batch.static_or("epsilon_fraction", float, defaults.EPSILON_FRACTION))
        ),
        BETA=cast(float, dynamic.get("beta", batch.static_or("beta", float, defaults.BETA))),
        H_REGULARIZATION=cast(Literal["adamw", "l2"], h_regularization),
        HIDDEN_WIDTH=batch.static_or("hidden_width", int, defaults.HIDDEN_WIDTH),
        ENV_NAME=batch.static("env_name", str),
        SEED=batch.static_or("seed", int, defaults.SEED),
    )


class QRCBatchTrainer:
    parameter_schema = QRC_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
    ) -> None:
        self._env_factory = env_factory

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static_config)
        integer_keys = {"learning_starts", "train_frequency"}
        dynamic_axes = {
            key: batch.dynamic(key, int if key in integer_keys else float)
            for key in self.parameter_schema.dynamic_keys
            if key in batch.dynamic_axes
        }

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = _config_from(batch, dynamic)
            _, metrics = run(
                QRCAgent(config),
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
                "algorithm": "qrc",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )


def make_qrc_batch_trainer(
    env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
) -> BatchTrainer:
    return QRCBatchTrainer(env_factory)
