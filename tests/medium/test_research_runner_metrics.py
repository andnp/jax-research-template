"""The runner records each declared metric by its shape: a curve, a per-run scalar or a batch scalar."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from experiment_definition.db import DatabaseManager, PlannedExecution
from experiment_definition.experiment import Experiment
from research_runner.config import traced
from research_runner.runner import execute_batch
from research_runner.types import ExperimentSpec


@dataclass(frozen=True)
class _Toy:
    offset: float = traced(0.0)
    mode: str = "a"


def _train(config: _Toy, key: jax.Array) -> tuple[object, dict[str, jax.Array]]:
    seed = key[1].astype(jnp.float32)
    curve = jnp.arange(6, dtype=jnp.float32) + 10 * seed
    return None, {
        "reward": curve,
        "diverging": jnp.where(curve > 2, jnp.nan, curve),
        "score": seed + 1.25,
        "unrecorded": curve,
    }


def _plan(tmp_path: Path, experiment: Experiment) -> tuple[ExperimentSpec[_Toy], PlannedExecution]:
    spec = ExperimentSpec(experiment, config=_Toy(), train=_train, results_root=tmp_path, capture_git=False)
    spec.sync(spec.db_path)
    with DatabaseManager(spec.db_path) as database:
        database.initialize()
        experiment_row = database.get_experiment(experiment.name)
        assert experiment_row is not None
        planned = database.plan_next_execution_batch(experiment_row.id, spec.executions_root)
        assert planned is not None
    return spec, planned


def _experiment(name: str) -> Experiment:
    return Experiment(name).add_parameter("seed", [0, 1])


def test_a_thinned_curve_is_stored_at_its_true_steps_for_each_run(tmp_path: Path) -> None:
    spec, planned = _plan(tmp_path, _experiment("curves").add_metric("reward", every=2))

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with sqlite3.connect(spec.metrics_db_path) as connection:
        rows = connection.execute(
            "SELECT run_id, seed_id, global_step, value FROM metrics ORDER BY run_id, global_step"
        ).fetchall()
    assert rows == [
        (1, 0, 0, 0.0), (1, 0, 2, 2.0), (1, 0, 4, 4.0),
        (2, 1, 0, 10.0), (2, 1, 2, 12.0), (2, 1, 4, 14.0),
    ]


def test_per_run_values_and_declared_runner_timings_are_stored_as_scalars(tmp_path: Path) -> None:
    experiment = _experiment("scalars").add_metric("score").add_metric("runner/execute_seconds")
    spec, planned = _plan(tmp_path, experiment)

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with sqlite3.connect(spec.metrics_db_path) as connection:
        rows = connection.execute(
            "SELECT scope, metric_name, run_id, seed_id, execution_id, value FROM scalar_metrics ORDER BY scope, run_id"
        ).fetchall()
    [timing, *scores] = rows
    assert timing[:5] == ("execution", "runner/execute_seconds", None, None, planned.execution_id)
    assert timing[5] > 0
    assert scores == [
        ("run", "score", 1, 0, planned.execution_id, 1.25),
        ("run", "score", 2, 1, planned.execution_id, 2.25),
    ]


def test_undeclared_metrics_are_not_stored(tmp_path: Path) -> None:
    spec, planned = _plan(tmp_path, _experiment("dropped").add_metric("score"))

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with sqlite3.connect(spec.metrics_db_path) as connection:
        curve_names = connection.execute("SELECT DISTINCT metric_name FROM metrics").fetchall()
        scalar_names = connection.execute("SELECT DISTINCT metric_name FROM scalar_metrics").fetchall()
    assert (curve_names, scalar_names) == ([], [("score",)])


def test_a_diverged_curve_completes_its_execution(tmp_path: Path) -> None:
    """A NaN curve is recorded as an outcome of a COMPLETED execution, not a failure."""
    spec, planned = _plan(tmp_path, Experiment("diverged").add_parameter("seed", [0]).add_metric("diverging"))

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with DatabaseManager(spec.db_path) as database:
        execution = database.get_execution(planned.execution_id)
    with sqlite3.connect(spec.metrics_db_path) as connection:
        values = connection.execute("SELECT value FROM metrics ORDER BY global_step").fetchall()
    assert execution is not None
    assert execution.status == "COMPLETED"
    assert values == [(0.0,), (1.0,), (2.0,), (None,), (None,), (None,)]


def test_execute_batch_records_runtime_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed execution records its thread count, batch width, wall time and peak memory."""
    monkeypatch.setenv("NPROC", "3")
    spec, planned = _plan(tmp_path, _experiment("telemetry"))

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with DatabaseManager(spec.db_path) as database:
        artifacts = database.get_execution_artifacts(planned.execution_id)
    assert artifacts is not None and artifacts.metadata_json is not None
    runtime = json.loads(artifacts.metadata_json)["runtime"]
    assert runtime["nproc"] == 3
    assert runtime["batch_runs"] == 2
    assert runtime["wall_seconds"] >= 0
    assert runtime["peak_rss_bytes"] > 0


def test_the_execution_records_how_each_metric_was_sampled(tmp_path: Path) -> None:
    """Sampling is not part of a run's identity, so the execution's artifact metadata is where it is kept."""
    experiment = _experiment("sampling").add_metric("reward", every=2, ema=0.9).add_metric("score")
    spec, planned = _plan(tmp_path, experiment)

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with DatabaseManager(spec.db_path) as database:
        artifacts = database.get_execution_artifacts(planned.execution_id)
    assert artifacts is not None and artifacts.metadata_json is not None
    expected = {"reward": {"every": 2, "ema": 0.9}, "score": {"every": 1, "ema": None}}
    assert json.loads(artifacts.metadata_json)["metrics"] == expected


def _recorded_metrics(spec: ExperimentSpec[_Toy], planned: PlannedExecution) -> dict[str, object]:
    with DatabaseManager(spec.db_path) as database:
        artifacts = database.get_execution_artifacts(planned.execution_id)
    assert artifacts is not None and artifacts.metadata_json is not None
    return json.loads(artifacts.metadata_json)["metrics"]


def test_a_conditional_metric_is_recorded_by_runs_that_match(tmp_path: Path) -> None:
    experiment = _experiment("matching").add_parameter("mode", ["a"])
    with experiment.when(mode="a"):
        experiment.add_metric("score")
    spec, planned = _plan(tmp_path, experiment)

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    with sqlite3.connect(spec.metrics_db_path) as connection:
        names = connection.execute("SELECT DISTINCT metric_name FROM scalar_metrics").fetchall()
    assert (names, list(_recorded_metrics(spec, planned))) == ([("score",)], ["score"])


def test_a_conditional_metric_is_dropped_for_runs_that_do_not_match(tmp_path: Path) -> None:
    experiment = _experiment("other-mode").add_parameter("mode", ["a"])
    with experiment.when(mode="b"):
        experiment.add_metric("emitted_only_by_mode_b")
    spec, planned = _plan(tmp_path, experiment)

    execute_batch(spec.db_path, planned, spec, capture_git=False)

    assert _recorded_metrics(spec, planned) == {}


def test_a_metric_conditioned_on_a_per_run_value_fails_when_a_batch_mixes_it(tmp_path: Path) -> None:
    experiment = _experiment("mixed")
    with experiment.when(seed=1):
        experiment.add_metric("score")
    spec, planned = _plan(tmp_path, experiment)

    with pytest.raises(ValueError, match="static parameters"):
        execute_batch(spec.db_path, planned, spec, capture_git=False)


def test_a_metric_conditioned_on_an_undeclared_parameter_fails_loudly(tmp_path: Path) -> None:
    experiment = _experiment("undeclared")
    with experiment.when(agent="sac"):
        experiment.add_metric("score")
    spec, planned = _plan(tmp_path, experiment)

    with pytest.raises(ValueError, match="conditioned on 'agent', which no run in this batch declares"):
        execute_batch(spec.db_path, planned, spec, capture_git=False)
