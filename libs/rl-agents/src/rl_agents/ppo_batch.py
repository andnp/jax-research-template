from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import jax
import jax.numpy as jnp
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.agent_protocol import AgentProtocol
from rl_components.batch import DynamicValue, run_batch
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


def _config_from(hyperparameters: Mapping[str, object]):
    return PPOConfig(
        LR=cast(float, hyperparameters["learning_rate"]),
        NUM_STEPS=cast(int, hyperparameters["num_steps"]),
        TOTAL_TIMESTEPS=cast(int, hyperparameters["total_timesteps"]),
        UPDATE_EPOCHS=cast(int, hyperparameters["update_epochs"]),
        NUM_MINIBATCHES=cast(int, hyperparameters["num_minibatches"]),
        CLIP_EPS=cast(float, hyperparameters["clip_eps"]),
        ENT_COEF=cast(float, hyperparameters["ent_coef"]),
        VF_COEF=cast(float, hyperparameters["vf_coef"]),
        MAX_GRAD_NORM=cast(float, hyperparameters["max_grad_norm"]),
        REWARD_SCALE=cast(float, hyperparameters["reward_scale"]),
        NORMALIZE_OBSERVATIONS=cast(bool, hyperparameters["normalize_observations"]),
        OBS_NORM_EPS=cast(float, hyperparameters["obs_norm_eps"]),
        OBS_NORM_CLIP=cast(float, hyperparameters["obs_norm_clip"]),
        ENV_NAME=cast(str, hyperparameters["env_name"]),
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
        static_config = dict(context.static_config)
        env_name = cast(str, static_config["env_name"])
        env, env_params = self._env_factory(env_name)
        axes = context.axes()
        seeds = tuple(_seed_of(point.hyperparameters["seed"], point.run_id) for point in context.points)
        dynamic_axes = {
            key: cast(Sequence[DynamicValue], values)
            for key, values in axes.items()
            if key != "seed"
        }
        unexpected = set(dynamic_axes) - self.parameter_schema.dynamic_keys
        if unexpected:
            raise ValueError(f"unsupported dynamic PPO axes: {sorted(unexpected)}")

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            hyperparameters: dict[str, object] = {**static_config, **dynamic}
            config = _config_from(hyperparameters)
            agent: AgentProtocol = PPOAgent(config)
            final_state, metrics = run(
                agent,
                env,
                key,
                steps=config.TOTAL_TIMESTEPS,
                gamma=cast(float, hyperparameters["gamma"]),
                env_params=env_params,
            )
            return final_state.agent_state.train_state.params, _held_episode_returns(
                cast(Mapping[str, jax.Array], metrics)
            )

        final_params, curves = run_batch(train_one, dynamic_axes, seeds)
        return ExecutionResult(
            metadata={
                "algorithm": "ppo",
                "env_name": static_config["env_name"],
                "seed_values": list(seeds),
            },
            metric_curves={"returned_episode_returns": curves.tolist()},
            artifacts={"final_policy_params": final_params},
        )


def _seed_of(value: ParameterValue, run_id: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Run {run_id} seed must be an int, got {value!r}.")
    return value


def make_ppo_batch_trainer(
    env_factory: Callable[[str], tuple[EnvProtocol[jax.Array, object, jax.Array, object], object]],
) -> BatchTrainer:
    return PPOBatchTrainer(env_factory)
