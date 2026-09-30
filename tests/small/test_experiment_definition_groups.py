"""Contract tests for run-batch group keys and the group planning filter."""

from datetime import datetime, timezone
from pathlib import Path

from experiment_definition.db import DatabaseManager


def _seed_groups(database: DatabaseManager) -> int:
    """Create four groups that differ only in ablation or algorithm version."""
    experiment_id = database.add_experiment("groups-test")
    component_id = database.add_component("algo", "ALGO")
    v1 = database.add_component_version(component_id, "algo-v1")
    v2 = database.add_component_version(component_id, "algo-v2")
    hyper_id = database.add_hyperparam_config({"condition_name": "baseline"})
    for version in (v1, v2):
        for ablation in ("full", "no-trick"):
            for seed in (0, 1):
                database.add_run(experiment_id, version, None, hyper_id, seed, ablation)
    return experiment_id


def test_group_keys_are_stable_and_distinct_per_group(tmp_path: Path) -> None:
    """Each group has its own key, and listing twice yields the same keys."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = _seed_groups(database)

        first = [b.group_key for b in database.list_unsatisfied_run_batches(experiment_id)]
        second = [b.group_key for b in database.list_unsatisfied_run_batches(experiment_id)]

        assert first == second
        assert len(set(first)) == 4


def test_group_filter_plans_only_that_group(tmp_path: Path) -> None:
    """Planning with a group key claims only that group's runs."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = _seed_groups(database)
        target = database.list_unsatisfied_run_batches(experiment_id)[2]

        filtered = database.list_unsatisfied_run_batches(experiment_id, group=target.group_key)
        planned = database.plan_next_execution_batch(
            experiment_id, tmp_path / "executions", group=target.group_key
        )

        assert [b.run_ids for b in filtered] == [target.run_ids]
        assert planned is not None
        assert planned.run_ids == target.run_ids
        assert database.plan_next_execution_batch(
            experiment_id, tmp_path / "executions", group=target.group_key
        ) is None
        assert database.plan_next_execution_batch(experiment_id, tmp_path / "executions") is not None


def test_groups_with_and_without_an_ablation_list_together(tmp_path: Path) -> None:
    """A run with no ablation and one with a named ablation form two listable groups."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = database.add_experiment("mixed-ablation")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline"})
        for ablation in (None, "no-trick"):
            database.add_run(experiment_id, None, None, hyper_id, 0, ablation)

        assert len(database.list_unsatisfied_run_batches(experiment_id)) == 2


def test_fail_group_executions_fails_only_that_groups_recent_running(tmp_path: Path) -> None:
    """Only RUNNING executions of the named group that started at or after the cutoff become FAILED.

    A dead worker's executions are exactly those, so older runs of the group and every other
    group's in-flight work must survive.
    """
    since = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    before, after = "2026-01-01T11:59:59+00:00", "2026-01-01T12:00:00.5+00:00"
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = _seed_groups(database)
        target, other = database.list_unsatisfied_run_batches(experiment_id)[:2]

        def running(run_id: int, start_time: str, status: str = "RUNNING") -> int:
            execution_id = database.add_execution()
            database.link_execution_run(execution_id, run_id)
            database.update_execution_status(execution_id, status, start_time=start_time)
            return execution_id

        dead = running(target.run_ids[0], after)
        older = running(target.run_ids[1], before)
        finished = running(target.run_ids[1], after, "COMPLETED")
        other_group = running(other.run_ids[0], after)

        def status(execution_id: int) -> str:
            execution = database.get_execution(execution_id)
            assert execution is not None
            return execution.status

        failed = database.fail_group_executions_started_since(experiment_id, target.group_key, since)
        statuses = {execution_id: status(execution_id) for execution_id in (dead, older, finished, other_group)}

    assert failed == [dead]
    assert statuses == {dead: "FAILED", older: "RUNNING", finished: "COMPLETED", other_group: "RUNNING"}
