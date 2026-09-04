"""Select an algorithm's batch trainer by the name an experiment sweeps.

An experiment that varies ``algorithm`` needs something to read that parameter and pick
a trainer. Nothing about that choice is project-specific, so it lives here rather than
being rewritten by every project that sweeps more than one algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.env_protocol import EnvProtocol

from rl_agents.dqn_batch import make_dqn_batch_trainer
from rl_agents.gac_batch import make_gac_batch_trainer
from rl_agents.ppo_batch import make_ppo_batch_trainer
from rl_agents.qrc_batch import make_qrc_batch_trainer
from rl_agents.sac_batch import make_sac_batch_trainer
from rl_agents.sac_rc_batch import make_sac_rc_batch_trainer
from rl_agents.td3_batch import make_td3_batch_trainer

EnvFactory = Callable[
    [Mapping[str, ParameterValue]],
    tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None],
]

ALGORITHM_PARAMETER = "algorithm"
"""The parameter naming which algorithm a batch of runs belongs to."""

ALGORITHM_TRAINERS: Mapping[str, Callable[[EnvFactory], BatchTrainer]] = {
    "dqn": make_dqn_batch_trainer,
    "gac": make_gac_batch_trainer,
    "ppo": make_ppo_batch_trainer,
    "qrc": make_qrc_batch_trainer,
    "sac": make_sac_batch_trainer,
    "sac-rc": make_sac_rc_batch_trainer,
    "td3": make_td3_batch_trainer,
}


class AlgorithmBatchTrainer:
    """Dispatch a batch of runs to the trainer named by its ``algorithm`` parameter."""

    def __init__(self, env_factory: EnvFactory, algorithms: Sequence[str]) -> None:
        algorithms = tuple(algorithms)
        if not algorithms:
            raise ValueError("an algorithm dispatch needs at least one algorithm")
        unknown = sorted(set(algorithms) - set(ALGORITHM_TRAINERS))
        if unknown:
            raise ValueError(f"no batch trainer registered for {unknown}; known: {sorted(ALGORITHM_TRAINERS)}")
        self._trainers = {name: ALGORITHM_TRAINERS[name](env_factory) for name in algorithms}

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


def make_algorithm_batch_trainer(env_factory: EnvFactory, algorithms: Sequence[str]) -> BatchTrainer:
    """Build a trainer that serves every algorithm in ``algorithms``."""
    return AlgorithmBatchTrainer(env_factory, algorithms)
