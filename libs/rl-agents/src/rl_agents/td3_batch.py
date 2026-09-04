from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import DynamicValue, run_batch
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
            "learning_starts",
            "gamma",
            "train_frequency",
            "tau",
            "policy_delay",
            "exploration_noise",
            "policy_noise",
            "noise_clip",
        }
    ),
    dynamic_keys=frozenset({"learning_rate"}),
)


class TD3BatchTrainer:
    parameter_schema = TD3_BATCH_PARAMETER_SCHEMA

    def __init__(
        self,
        env_factory: Callable[[str], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]],
    ) -> None:
        self._env_factory = env_factory

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        static_config = dict(context.static_config)
        env_name = cast(str, static_config["env_name"])
        env, env_params = self._env_factory(env_name)
        axes = context.axes()
        seeds = tuple(_seed_of(point.hyperparameters["seed"], point.run_id) for point in context.points)
        dynamic_axes = {
            key: cast(list[DynamicValue], values)
            for key, values in axes.items()
            if key != "seed"
        }
        unexpected = set(dynamic_axes) - self.parameter_schema.dynamic_keys
        if unexpected:
            raise ValueError(f"unsupported dynamic TD3 axes: {sorted(unexpected)}")

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = TD3Config(
                LR=cast(float, dynamic["learning_rate"]),
                BUFFER_SIZE=cast(int, static_config["buffer_size"]),
                BATCH_SIZE=cast(int, static_config["batch_size"]),
                TOTAL_TIMESTEPS=cast(int, static_config["total_timesteps"]),
                LEARNING_STARTS=cast(int, static_config["learning_starts"]),
                TRAIN_FREQUENCY=cast(int, static_config["train_frequency"]),
                TAU=cast(float, static_config["tau"]),
                POLICY_DELAY=cast(int, static_config["policy_delay"]),
                EXPLORATION_NOISE=cast(float, static_config["exploration_noise"]),
                POLICY_NOISE=cast(float, static_config["policy_noise"]),
                NOISE_CLIP=cast(float, static_config["noise_clip"]),
                ENV_NAME=cast(str, static_config["env_name"]),
                SEED=0,
            )
            agent = TD3Agent(config)
            _, metrics = run(
                agent,
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=cast(float, static_config["gamma"]),
                env_params=env_params,
            )
            return {
                "reward": metrics["loop/reward"],
                "critic_loss": metrics["critic_loss"],
                "actor_loss": metrics["actor_loss"],
            }

        metrics = run_batch(train_one, dynamic_axes, seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "td3",
                "env_name": static_config["env_name"],
                "seed_values": list(seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )


def _seed_of(value: ParameterValue, run_id: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Run {run_id} seed must be an int, got {value!r}.")
    return value


def make_td3_batch_trainer(
    env_factory: Callable[[str], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]],
) -> BatchTrainer:
    return TD3BatchTrainer(env_factory)
