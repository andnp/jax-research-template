"""Contract tests for validated experiment database records."""

from pathlib import Path

from experiment_definition.db import DatabaseManager


def test_database_reads_return_named_records(tmp_path: Path) -> None:
    """Return typed records with nullable provenance preserved at the database boundary."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        experiment_id = database.add_experiment("records-test")
        hyper_id = database.add_hyperparam_config({"condition_name": "baseline"})
        run_id = database.add_run(experiment_id, None, None, hyper_id, 0)

        experiment = database.list_experiments()[0]
        run = database.list_runs(experiment.id)[0]

        assert experiment.id == experiment_id
        assert run.id == run_id
        assert run.env_version_id is None
        assert experiment.model_dump()["name"] == "records-test"
        assert run.model_dump()["id"] == run_id


def test_database_reads_component_provenance(tmp_path: Path) -> None:
    """Return joined component identity as one validated provenance record."""
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        component_id = database.add_component("ppo", "ALGO")
        version_id = database.add_component_version(component_id, "algo-v1")

        provenance = database.get_component_version_provenance(version_id)

        assert provenance is not None
        assert provenance.component_id == component_id
        assert provenance.component_name == "ppo"
        assert provenance.version_id == version_id
