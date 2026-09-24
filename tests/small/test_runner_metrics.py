import json
import sqlite3
from pathlib import Path

from experiment_definition import Experiment
from experiment_definition.db import DatabaseManager
from research_runner import ExecutionContext, ExecutionResult, execute_batch


def test_execute_batch_persists_result_curves(tmp_path: Path) -> None:
    """
    Verify the runner persists trainer metric curves with run and seed identity.
    """
    database_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    experiment = Experiment("metrics-test")
    experiment.add_parameter("seed", [0, 1])
    experiment.add_metric("reward", frequency="per_update")
    experiment.sync(database_path)

    with DatabaseManager(database_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(experiment.name)
        assert experiment_row is not None
        planned = database.plan_next_execution_batch(experiment_row.id, tmp_path / "executions")
        assert planned is not None

    def train(context: ExecutionContext) -> ExecutionResult:
        assert len(context.points) == 2
        return ExecutionResult(
            metric_curves={"reward": ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))}
        )

    execute_batch(
        database_path,
        planned,
        train,
        metrics_db_path=metrics_path,
        capture_git=False,
        metric_subsample_factor=2,
    )

    with sqlite3.connect(metrics_path) as connection:
        rows = connection.execute(
            "SELECT run_id, seed_id, global_step, value "
            "FROM metrics ORDER BY run_id, global_step"
        ).fetchall()

    assert rows == [(1, 0, 0, 1.0), (1, 0, 2, 3.0), (2, 1, 0, 4.0), (2, 1, 2, 6.0)]


def test_execute_batch_persists_result_scalars(tmp_path: Path) -> None:
    """
    Verify the runner persists scalar values at run and execution scope.
    """
    database_path = tmp_path / "experiments.sqlite"
    metrics_path = tmp_path / "metrics.sqlite"
    experiment = Experiment("scalar-test")
    experiment.add_parameter("seed", [0, 1])
    experiment.sync(database_path)

    with DatabaseManager(database_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(experiment.name)
        assert experiment_row is not None
        experiment_id = experiment_row.id
        planned = database.plan_next_execution_batch(experiment_id, tmp_path / "executions")
        assert planned is not None

    def train(context: ExecutionContext) -> ExecutionResult:
        assert len(context.points) == 2
        return ExecutionResult(
            run_scalars={"score": (1.25, 2.5)},
            execution_scalars={"batch_score": 1.875},
        )

    execute_batch(database_path, planned, train, metrics_db_path=metrics_path, capture_git=False)

    with sqlite3.connect(metrics_path) as connection:
        rows = connection.execute(
            "SELECT scope, metric_name, experiment_id, run_id, seed_id, execution_id, value "
            "FROM scalar_metrics ORDER BY scope, metric_name, run_id"
        ).fetchall()

    assert rows == [
        ("execution", "batch_score", experiment_id, None, None, planned.execution_id, 1.875),
        ("run", "score", experiment_id, 1, 0, planned.execution_id, 1.25),
        ("run", "score", experiment_id, 2, 1, planned.execution_id, 2.5),
    ]


def test_execute_batch_completes_diverged_curves(tmp_path: Path) -> None:
    """
    Verify a NaN curve is recorded as an outcome of a COMPLETED execution, not a failure.
    """
    database_path = tmp_path / "experiments.sqlite"
    experiment = Experiment("diverged-test")
    experiment.add_parameter("seed", [0])
    experiment.sync(database_path)
    with DatabaseManager(database_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(experiment.name)
        assert experiment_row is not None
        planned = database.plan_next_execution_batch(experiment_row.id, tmp_path / "executions")
        assert planned is not None

    def train(context: ExecutionContext) -> ExecutionResult:
        return ExecutionResult(metric_curves={"reward": ((1.0, float("nan")),)})

    execute_batch(database_path, planned, train, metrics_db_path=tmp_path / "metrics.sqlite", capture_git=False)

    with DatabaseManager(database_path) as database:
        execution = database.get_execution(planned.execution_id)
    with sqlite3.connect(tmp_path / "metrics.sqlite") as connection:
        values = connection.execute("SELECT value FROM metrics ORDER BY global_step").fetchall()
    assert execution is not None
    assert execution.status == "COMPLETED"
    assert values == [(1.0,), (None,)]


def test_execute_batch_persists_result_artifacts(tmp_path: Path) -> None:
    """
    Verify the runner owns artifact storage and records its URI in the manifest.
    """
    database_path = tmp_path / "experiments.sqlite"
    experiment = Experiment("artifact-test")
    experiment.add_parameter("seed", [0])
    experiment.sync(database_path)

    with DatabaseManager(database_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(experiment.name)
        assert experiment_row is not None
        planned = database.plan_next_execution_batch(experiment_row.id, tmp_path / "executions")
        assert planned is not None

    def train(context: ExecutionContext) -> ExecutionResult:
        assert len(context.points) == 1
        return ExecutionResult(artifacts={"summary": {"value": 3}})

    execute_batch(database_path, planned, train, capture_git=False)

    manifest = json.loads(Path(planned.manifest_path).read_text(encoding="utf-8"))
    assert manifest["metadata"]["artifacts"]["summary"].startswith("research://")
    assert list((Path(planned.root_path) / "artifacts").rglob("summary_v1.pkl"))
