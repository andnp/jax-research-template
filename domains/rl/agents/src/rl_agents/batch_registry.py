"""Select an algorithm's batch trainer by the name an experiment sweeps.

An experiment that varies ``algorithm`` needs something to read that parameter and pick
a trainer. Nothing about that choice is project-specific, so it lives here rather than
being rewritten by every project that sweeps more than one algorithm.

Each registered algorithm is an :class:`~rl_agents.batch_trainer.AlgorithmSpec`, not a
hand-written trainer factory: :func:`~rl_agents.batch_trainer.make_batch_trainer` builds
the actual :class:`~research_runner.BatchTrainer` from the spec plus the env factory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.env_protocol import EnvProtocol
from rl_components.types import PPOConfig

from rl_agents.batch_trainer import AlgorithmSpec, make_batch_trainer
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.greedy_ac import GACAgent, GACConfig, GACHypers
from rl_agents.ppo import PPOAgent, PPOHypers
from rl_agents.qrc import QRCAgent, QRCConfig, QRCHypers
from rl_agents.replay_q_agent import ReplayQHypers
from rl_agents.sac import SACAgent, SACConfig, SACHypers
from rl_agents.sac_rc import SACRCAgent, SACRCConfig, SACRCHypers
from rl_agents.td3 import TD3Agent, TD3Config, TD3Hypers

EnvFactory = Callable[
    [Mapping[str, ParameterValue]],
    tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None],
]

ALGORITHM_PARAMETER = "algorithm"
"""The parameter naming which algorithm a batch of runs belongs to."""

ALGORITHM_TRAINERS: Mapping[str, AlgorithmSpec] = {
    "dqn": AlgorithmSpec("dqn", DQNAgent, DQNConfig, ReplayQHypers),
    "gac": AlgorithmSpec("gac", GACAgent, GACConfig, GACHypers),
    "ppo": AlgorithmSpec("ppo", PPOAgent, PPOConfig, PPOHypers),
    "qrc": AlgorithmSpec("qrc", QRCAgent, QRCConfig, QRCHypers),
    "sac": AlgorithmSpec("sac", SACAgent, SACConfig, SACHypers),
    "sac-rc": AlgorithmSpec("sac-rc", SACRCAgent, SACRCConfig, SACRCHypers),
    "td3": AlgorithmSpec("td3", TD3Agent, TD3Config, TD3Hypers),
}


class AlgorithmBatchTrainer:
    """Dispatch a batch of runs to the trainer named by its ``algorithm`` parameter."""

    def __init__(
        self,
        env_factory: EnvFactory,
        algorithms: Sequence[str],
        *,
        extra_specs: Mapping[str, AlgorithmSpec] = {},
        report_returned_episode_returns: bool = False,
    ) -> None:
        algorithms = tuple(algorithms)
        if not algorithms:
            raise ValueError("an algorithm dispatch needs at least one algorithm")
        specs = {**ALGORITHM_TRAINERS, **extra_specs}
        unknown = sorted(set(algorithms) - set(specs))
        if unknown:
            raise ValueError(f"no batch trainer registered for {unknown}; known: {sorted(specs)}")
        self._trainers = {
            name: make_batch_trainer(
                specs[name], env_factory, report_returned_episode_returns=report_returned_episode_returns
            )
            for name in algorithms
        }

    @property
    def parameter_schema(self) -> ParameterSchema:
        """The keys every member can batch, which is their intersection.

        One schema is read per experiment, before any batch is known, and it decides
        which parameters the runner may vary within a single batch. A key only some
        members accept would therefore be promised for runs that cannot honour it, and
        the selected member would reject the batch it was handed.
        """
        return ParameterSchema(
            dynamic_keys=frozenset.intersection(
                *(trainer.parameter_schema.dynamic_keys for trainer in self._trainers.values())
            )
        )

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        name = context.static_config.get(ALGORITHM_PARAMETER)
        if not isinstance(name, str):
            raise ValueError(f"{ALGORITHM_PARAMETER!r} must be text naming one algorithm, got {name!r}")
        try:
            trainer = self._trainers[name]
        except KeyError as error:
            raise ValueError(f"algorithm {name!r} is not one of {sorted(self._trainers)}") from error
        return trainer(context)


def make_algorithm_batch_trainer(
    env_factory: EnvFactory,
    algorithms: Sequence[str],
    *,
    extra_specs: Mapping[str, AlgorithmSpec] = {},
    report_returned_episode_returns: bool = False,
) -> BatchTrainer:
    """Build a trainer that serves every algorithm in ``algorithms``.

    ``extra_specs`` lets a project register its own :class:`AlgorithmSpec`\\ s (e.g. a
    project-local algorithm variant) alongside the built-in registry without forking it;
    a name present in both wins from ``extra_specs``.
    """
    return AlgorithmBatchTrainer(
        env_factory,
        algorithms,
        extra_specs=extra_specs,
        report_returned_episode_returns=report_returned_episode_returns,
    )
