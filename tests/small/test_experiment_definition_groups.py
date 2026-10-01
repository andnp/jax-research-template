"""Contract tests for run-batch group keys and the group planning filter."""

from pathlib import Path

from experiment_definition.db import DatabaseManager


def _seed_groups(database: DatabaseManager, seeds: tuple[int, ...] = (0, 1)) -> int:
    """Create four groups that differ only in ablation or algorithm version."""
    experiment_id = database.add_experiment("groups-test")
    component_id = database.add_component("algo", "ALGO")
    v1 = database.add_component_version(component_id, "algo-v1")
    v2 = database.add_component_version(component_id, "algo-v2")
    hyper_id = database.add_hyperparam_config({"condition_name": "baseline"})
    for version in (v1, v2):
        for ablation in ("full", "no-trick"):
            for seed in seeds:
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

def test_batch_planning_with_a_group_plans_every_batch_of_that_group_only(tmp_path: Path) -> None:
    """Planning all batches with a group key plans that group's batches, in order, and no other group's."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = _seed_groups(database, seeds=(0, 1, 2))
        target = database.list_unsatisfied_run_batches(experiment_id)[0]
        expected = [b.run_ids for b in database.list_unsatisfied_run_batches(experiment_id, max_runs_per_batch=2, group=target.group_key)]

        planned = database.plan_experiment_execution_batches(
            experiment_id, tmp_path / "executions", max_runs_per_batch=2, group=target.group_key
        )

        assert len(expected) == 2
        assert [p.run_ids for p in planned] == expected
        assert {run_id for p in planned for run_id in p.run_ids} == set(target.run_ids)
        remaining = database.list_unsatisfied_run_batches(experiment_id)
        assert target.group_key not in {b.group_key for b in remaining}
        assert len(remaining) == 3


def test_groups_with_and_without_an_ablation_list_together(tmp_path: Path) -> None:
    """A run with no ablation and one with a named ablation form two listable groups."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = database.add_experiment("mixed-ablation")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline"})
        for ablation in (None, "no-trick"):
            database.add_run(experiment_id, None, None, hyper_id, 0, ablation)

        assert len(database.list_unsatisfied_run_batches(experiment_id)) == 2


def test_all_traced_sweep_batches_together_in_chunks(tmp_path: Path) -> None:
    """Runs differing only in traced values share a group, chunked by max_runs_per_batch."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = database.add_experiment("traced-test")
        version = database.add_component_version(database.add_component("algo", "ALGO"), "algo-v1")
        zone = {"static_keys": [], "dynamic_keys": ["agent.learning_rate"]}
        for learning_rate in (0.1, 0.2, 0.3):
            hyper_id = database.add_hyperparam_config({"agent.learning_rate": learning_rate})
            for seed in (0, 1):
                database.add_run(experiment_id, version, None, hyper_id, seed, vmap_zone=zone)

        whole = database.list_unsatisfied_run_batches(experiment_id)
        chunked = database.list_unsatisfied_run_batches(experiment_id, max_runs_per_batch=4)

        assert [len(b.run_ids) for b in whole] == [6]
        assert [len(b.run_ids) for b in chunked] == [4, 2]
