"""Contract tests for opening results databases written by an older schema."""

import sqlite3
from pathlib import Path

from experiment_definition.db import DatabaseManager

_ZONE_JSON = '{"dynamic_keys": ["lr"], "static_keys": ["arch"]}'

_LEGACY_DDL = [
    "CREATE TABLE Experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, "
    "description TEXT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))",
    "CREATE TABLE HyperparamConfigs (id INTEGER PRIMARY KEY AUTOINCREMENT, hash TEXT NOT NULL UNIQUE, "
    "json_blob TEXT NOT NULL, vmap_zone_json TEXT)",
    "CREATE TABLE Runs (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER NOT NULL, "
    "algo_version_id INTEGER, env_version_id INTEGER, hyper_id INTEGER NOT NULL, seed INTEGER NOT NULL, "
    "ablation TEXT, UNIQUE(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation))",
]


def _write_legacy_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as con:
        for ddl in _LEGACY_DDL:
            con.execute(ddl)
        con.execute("INSERT INTO Experiments(name) VALUES ('legacy')")
        for hyper_id, lr in ((1, 1e-3), (2, 3e-4)):
            con.execute(
                "INSERT INTO HyperparamConfigs(hash, json_blob, vmap_zone_json) VALUES (?, ?, ?)",
                (f"h{hyper_id}", f'{{"arch": "mlp", "lr": {lr}}}', _ZONE_JSON),
            )
            con.execute("INSERT INTO Runs(experiment_id, hyper_id, seed) VALUES (1, ?, 0)", (hyper_id,))


def test_legacy_database_keeps_its_batches_after_upgrade(tmp_path: Path) -> None:
    """Runs of a pre-upgrade database inherit their config's zone, so they still batch together.

    Results databases are durable; opening one with a newer core must not re-split its
    pending runs or change their group keys.
    """
    db_path = tmp_path / "experiments.sqlite"
    _write_legacy_database(db_path)

    with DatabaseManager(db_path) as database:
        database.initialize()
        zones = {row[0] for row in database.conn.execute("SELECT vmap_zone_json FROM Runs")}
        batches = database.list_unsatisfied_run_batches(1)

    assert zones == {_ZONE_JSON}
    assert [(batch.static_config, len(batch.run_ids)) for batch in batches] == [({"arch": "mlp"}, 2)]


def test_executions_gain_an_owner_column_on_initialize(tmp_path: Path) -> None:
    """A database written before ownership gets a NULL owner on its executions, and re-initializing is harmless."""
    db_path = tmp_path / "experiments.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE Executions (id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL DEFAULT 'PENDING', "
            "hostname TEXT, start_time TEXT, end_time TEXT, git_commit TEXT, git_diff_blob TEXT, jax_config_json TEXT)"
        )
        con.execute("INSERT INTO Executions(hostname) VALUES ('legacy-host')")

    with DatabaseManager(db_path) as database:
        database.initialize()
        database.initialize()
        execution = database.get_execution(1)

    assert execution is not None
    assert (execution.hostname, execution.owner) == ("legacy-host", None)


def test_database_with_a_manifest_path_column_still_records_artifacts(tmp_path: Path) -> None:
    """Databases written while artifacts had a manifest path keep their column, which is no longer read or written."""
    db_path = tmp_path / "experiments.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE ExecutionArtifacts (execution_id INTEGER PRIMARY KEY, root_path TEXT NOT NULL, "
            "manifest_path TEXT, metadata_json TEXT, "
            "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))"
        )
        con.execute("INSERT INTO ExecutionArtifacts(execution_id, root_path, manifest_path) VALUES (1, '/old', '/old/manifest.json')")

    with DatabaseManager(db_path) as database:
        database.initialize()
        assert database.add_execution() == 1
        before = database.get_execution_artifacts(1)
        database.record_execution_artifacts(1, "/new", metadata={"metrics_db_path": "m.sqlite"})
        after = database.get_execution_artifacts(1)

    assert before is not None and before.root_path == "/old"
    assert after is not None and after.root_path == "/new"
    assert after.metadata_json == '{"metrics_db_path": "m.sqlite"}'
