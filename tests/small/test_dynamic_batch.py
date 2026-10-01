from __future__ import annotations

from pathlib import Path

import pytest
from research_runner.types import BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema, RunPoint


class _Trainer:
    parameter_schema = ParameterSchema(dynamic_keys=frozenset({"learning_rate"}))

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        del context
        return ExecutionResult()


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
