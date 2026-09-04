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
