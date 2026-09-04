from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
from experiment_definition import Experiment
from research_runner import ExecutionContext, ExecutionResult, ExperimentSpec, ParameterSchema
from rl_components.batch import run_batch


class _Trainer:
    parameter_schema = ParameterSchema(
        static_keys=frozenset({"env_name"}),
        dynamic_keys=frozenset({"learning_rate"}),
    )

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

    with sqlite3.connect(spec.db_path) as connection:
        parameter_rows = connection.execute(
            "SELECT name, is_static FROM ParameterSpecs ORDER BY name"
        ).fetchall()
        zone_rows = connection.execute(
            "SELECT vmap_zone_json FROM HyperparamConfigs"
        ).fetchall()

    assert parameter_rows == [("env_name", 1), ("learning_rate", 0), ("seed", 0)]
    zones = {row[0] for row in zone_rows}
    assert zones == {
        '{"dynamic_keys": ["learning_rate"], "static_keys": ["env_name"]}'
    }
