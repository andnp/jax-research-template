from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from experiment_definition import Experiment
from experiment_definition.db import DatabaseManager
from research_runner import BoundBatch, ExecutionContext, ExecutionResult, ExperimentSpec, ParameterSchema, RunPoint
from rl_components.batch import run_batch


class _Trainer:
    parameter_schema = ParameterSchema(dynamic_keys=frozenset({"learning_rate"}))

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        del context
        return ExecutionResult()


def test_run_batch_maps_dynamic_values_with_seed_keys() -> None:
    """
    Verify the shared batch adapter maps dynamic values and independent RNG keys.
    """
    def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array) -> jax.Array:
        return dynamic["learning_rate"] + jnp.asarray(key[0], dtype=jnp.float32) * 0.0

    values = run_batch(train_one, {"learning_rate": (1e-4, 1e-3)}, (0, 1))

    assert values.shape == (2,)
    assert jnp.allclose(values, jnp.array([1e-4, 1e-3]))


def test_context_binds_typed_static_and_dynamic_values() -> None:
    """
    Verify trainers receive validated native values and ordered dynamic axes.
    """
    context = ExecutionContext(
        execution_id=1,
        experiment_id=1,
        experiment_name="schema-test",
        static_config={"env_name": "cartpole"},
        points=(
            RunPoint(1, {"env_name": "cartpole", "learning_rate": 1e-4, "seed": 3}),
            RunPoint(2, {"env_name": "cartpole", "learning_rate": 1e-3, "seed": 4}),
        ),
        execution_root=Path("/tmp/execution"),
        metrics_db_path=Path("/tmp/metrics.sqlite"),
    )

    batch = context.bind(_Trainer.parameter_schema)

    assert isinstance(batch, BoundBatch)
    assert batch.static("env_name", str) == "cartpole"
    assert batch.dynamic("learning_rate", float) == (1e-4, 1e-3)
    assert batch.seeds == (3, 4)


def test_run_point_get_extracts_typed_hyperparameter() -> None:
    """
    Verify RunPoint.get() coerces and validates a value without binding a schema.
    """
    point = RunPoint(1, {"learning_rate": 1e-4, "seed": 3})

    assert point.get("learning_rate", float) == 1e-4
    assert point.get("seed", int) == 3
    with pytest.raises(TypeError, match="learning_rate"):
        point.get("learning_rate", int)
    with pytest.raises(ValueError, match="missing"):
        point.get("momentum", float)


def test_execution_context_get_static_extracts_typed_value() -> None:
    """
    Verify ExecutionContext.get_static() reads static_config without binding a schema.
    """
    context = ExecutionContext(
        execution_id=1,
        experiment_id=1,
        experiment_name="schema-test",
        static_config={"env_name": "cartpole"},
        points=(),
        execution_root=Path("/tmp/execution"),
        metrics_db_path=Path("/tmp/metrics.sqlite"),
    )

    assert context.get_static("env_name", str) == "cartpole"
    with pytest.raises(ValueError, match="missing"):
        context.get_static("seed", int)


def test_context_rejects_unregistered_dynamic_values() -> None:
    """
    Verify a trainer cannot silently ignore an unsupported dynamic parameter.
    """
    context = ExecutionContext(
        execution_id=1,
        experiment_id=1,
        experiment_name="schema-test",
        static_config={"env_name": "cartpole"},
        points=(RunPoint(1, {"env_name": "cartpole", "momentum": 0.9, "seed": 3}),),
        execution_root=Path("/tmp/execution"),
        metrics_db_path=Path("/tmp/metrics.sqlite"),
    )

    with pytest.raises(ValueError, match="momentum"):
        context.bind(_Trainer.parameter_schema)


def test_experiment_spec_applies_trainer_schema(tmp_path: Path) -> None:
    """
    Verify planning derives static and dynamic zones from the trainer contract.
    """
    experiment = Experiment("schema-test")
    experiment.add_parameter("env_name", ["cartpole"])
    experiment.add_parameter("learning_rate", [1e-4, 1e-3])
    experiment.add_parameter("seed", [0, 1])
    spec = ExperimentSpec(
        experiment=experiment,
        train_fn=_Trainer(),
        results_root=tmp_path,
    )

    spec.sync(spec.db_path)

    with DatabaseManager(spec.db_path) as database:
        experiment_row = database.get_experiment(experiment.name)
        assert experiment_row is not None
        runs = database.list_runs(experiment_row.id)
        zones = set()
        for run in runs:
            config = database.get_hyperparam_config(run.hyper_id)
            assert config is not None
            assert config.vmap_zone is not None
            zones.add((tuple(config.vmap_zone["static_keys"]), tuple(config.vmap_zone["dynamic_keys"])))

    assert zones == {(("env_name",), ("learning_rate",))}
