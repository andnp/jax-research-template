"""Contract tests for run-batch group keys and the group planning filter."""

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
