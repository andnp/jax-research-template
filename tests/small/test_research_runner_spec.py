"""A spec's sync batches each run by the traced fields its parameters set."""

from dataclasses import dataclass
from pathlib import Path

import jax
from experiment_definition.db import DatabaseManager
from experiment_definition.experiment import Experiment
from research_runner.config import traced
from research_runner.types import ExperimentSpec


@dataclass(frozen=True)
class _TracedRate:
    LEARNING_RATE: float = traced(1e-3)


@dataclass(frozen=True)
class _StaticRate:
    LEARNING_RATE: float = 1e-3


@dataclass(frozen=True)
class _Agents:
    agent: _TracedRate | _StaticRate = _TracedRate()


@dataclass(frozen=True)
class _Zoned:
    env_name: str = "cartpole"
    learning_rate: float = traced(1e-3)


def _train(config: object, key: jax.Array) -> tuple[object, dict[str, jax.Array]]:
    return None, {}


def _zones(db_path: Path, name: str) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    with DatabaseManager(db_path) as database:
        experiment_row = database.get_experiment(name)
        assert experiment_row is not None
        zones = set()
        for run in database.list_runs(experiment_row.id):
            assert run.vmap_zone is not None
            zones.add((tuple(run.vmap_zone["static_keys"]), tuple(run.vmap_zone["dynamic_keys"])))
    return zones


def test_parameters_setting_traced_fields_vary_within_a_batch(tmp_path: Path) -> None:
    experiment = Experiment("zones")
    experiment.add_parameter("env_name", ["cartpole"])
    experiment.add_parameter("learning_rate", [1e-4, 1e-3])
    experiment.add_parameter("seed", [0, 1])
    spec = ExperimentSpec(experiment, config=_Zoned(), train=_train, results_root=tmp_path)

    spec.sync(spec.db_path)

    assert _zones(spec.db_path, experiment.name) == {(("env_name",), ("learning_rate",))}


def test_each_run_is_batched_by_the_traced_fields_of_its_own_choice(tmp_path: Path) -> None:
    experiment = Experiment("choices")
    experiment.add_parameter("agent", ["traced", "static"])
    experiment.add_parameter("agent.LEARNING_RATE", [1e-4, 1e-3])
    experiment.add_parameter("seed", [0])
    choices = {"agent": {"traced": _TracedRate(), "static": _StaticRate()}}
    spec = ExperimentSpec(experiment, config=_Agents(), choices=choices, train=_train, results_root=tmp_path)

    spec.sync(spec.db_path)

    assert _zones(spec.db_path, experiment.name) == {
        (("agent",), ("agent.LEARNING_RATE",)),
        (("agent", "agent.LEARNING_RATE"), ()),
    }
