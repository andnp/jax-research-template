from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run

from rl_agents.greedy_ac import GACAgent, GACConfig

GAC_BATCH_PARAMETER_SCHEMA = ParameterSchema(
    dynamic_keys=frozenset(
        {
            "learning_rate",
            "actor_learning_rate",
            "gamma",
            "learning_starts",
            "train_frequency",
            "entropy_weight",
            "training_sigma_min",
            "inference_sigma_min",
            "h_regularization",
        }
    ),
)


def _config_from(batch: BoundBatch, dynamic: Mapping[str, jax.Array]):
    defaults = GACConfig()
    return GACConfig(
        LEARNING_RATE=cast(float, dynamic.get("learning_rate", batch.static_or("learning_rate", float, defaults.LEARNING_RATE))),
        ACTOR_LEARNING_RATE=cast(
            float, dynamic.get("actor_learning_rate", batch.static_or("actor_learning_rate", float, defaults.ACTOR_LEARNING_RATE))
        ),
        BUFFER_SIZE=batch.static_or("buffer_size", int, defaults.BUFFER_SIZE),
        BATCH_SIZE=batch.static_or("batch_size", int, defaults.BATCH_SIZE),
        TOTAL_TIMESTEPS=batch.static_or("total_timesteps", int, defaults.TOTAL_TIMESTEPS),
        LEARNING_STARTS=cast(
            int, dynamic.get("learning_starts", batch.static_or("learning_starts", int, defaults.LEARNING_STARTS))
        ),
        TRAIN_FREQUENCY=cast(
            int, dynamic.get("train_frequency", batch.static_or("train_frequency", int, defaults.TRAIN_FREQUENCY))
        ),
        TAU=batch.static_or("tau", float, defaults.TAU),
        NUM_SAMPLES=batch.static_or("num_samples", int, defaults.NUM_SAMPLES),
        ACTOR_PERCENTILE=batch.static_or("actor_percentile", float, defaults.ACTOR_PERCENTILE),
        UNIFORM_WEIGHT=batch.static_or("uniform_weight", float, defaults.UNIFORM_WEIGHT),
        ENTROPY_WEIGHT=cast(
            float, dynamic.get("entropy_weight", batch.static_or("entropy_weight", float, defaults.ENTROPY_WEIGHT))
        ),
        NUM_RAND_ACTIONS=batch.static_or("num_rand_actions", int, defaults.NUM_RAND_ACTIONS),
        TRAINING_SIGMA_MIN=cast(
            float,
            dynamic.get("training_sigma_min", batch.static_or("training_sigma_min", float, defaults.TRAINING_SIGMA_MIN)),
        ),
        INFERENCE_SIGMA_MIN=cast(
            float,
            dynamic.get("inference_sigma_min", batch.static_or("inference_sigma_min", float, defaults.INFERENCE_SIGMA_MIN)),
        ),
        H_REGULARIZATION=cast(
            float, dynamic.get("h_regularization", batch.static_or("h_regularization", float, defaults.H_REGULARIZATION))
        ),
        HIDDEN_SIZE=batch.static_or("hidden_size", int, defaults.HIDDEN_SIZE),
        ENV_NAME=batch.static("env_name", str),
        SEED=batch.static_or("seed", int, defaults.SEED),
    )


class GACBatchTrainer:
    parameter_schema = GAC_BATCH_PARAMETER_SCHEMA

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
                GACAgent(config),
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=cast(float, dynamic.get("gamma", batch.static_or("gamma", float, 0.99))),
                env_params=env_params,
            )
            return {
                "reward": metrics["loop/reward"],
                "critic_loss": metrics["critic_loss"],
                "actor_loss": metrics["actor_loss"],
                "critic_grad_norm": metrics["critic_grad_norm"],
                "actor_grad_norm": metrics["actor_grad_norm"],
            }

        metrics = run_batch(train_one, dynamic_axes, batch.seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "gac",
                "env_name": batch.static("env_name", str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )


def make_gac_batch_trainer(
    env_factory: Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]],
) -> BatchTrainer:
    return GACBatchTrainer(env_factory)
