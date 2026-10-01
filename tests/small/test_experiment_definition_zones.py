"""Contract tests for vmap-zone batching when experiments share a database."""

import sqlite3
from pathlib import Path

from experiment_definition.db import DatabaseManager
from experiment_definition.experiment import Experiment


def _grid(name: str):
    experiment = Experiment(name)
    experiment.add_parameter("seed", [0, 1])
    experiment.add_parameter("arch", ["mlp"])
    experiment.add_parameter("lr", [1e-3, 3e-4])
    return experiment


def _batch_shapes(database: DatabaseManager, name: str):
    experiment_row = database.get_experiment(name)
    assert experiment_row is not None
    return sorted(
        ((dict(batch.static_config), len(batch.run_ids)) for batch in database.list_unsatisfied_run_batches(experiment_row.id)),
        key=repr,
    )


def test_experiments_sharing_a_grid_batch_by_their_own_zones(tmp_path: Path) -> None:
    """Two experiments with identical configs but different static splits each batch by their own split.

    Their runs share hyperparameter-config rows, so a zone stored per config would let the
    last sync decide how the other experiment is batched.
    """
    db_path = tmp_path / "experiments.sqlite"
    _grid("vmapped").sync(db_path, static_names=lambda _: {"arch"})
    _grid("per-lr").sync(db_path, static_names=lambda _: {"arch", "lr"})

    with DatabaseManager(db_path) as database:
        vmapped = _batch_shapes(database, "vmapped")
        per_lr = _batch_shapes(database, "per-lr")

    assert vmapped == [({"arch": "mlp"}, 4)]
    assert per_lr == [({"arch": "mlp", "lr": 0.0003}, 2), ({"arch": "mlp", "lr": 0.001}, 2)]


def _downgrade_to_config_zones(db_path: Path, zone_json: str) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("ALTER TABLE HyperparamConfigs ADD COLUMN vmap_zone_json TEXT")
        con.execute("UPDATE HyperparamConfigs SET vmap_zone_json = ?", (zone_json,))
        con.execute("ALTER TABLE Runs DROP COLUMN vmap_zone_json")


def test_resync_heals_zones_lost_when_upgrading_a_shared_database(tmp_path: Path) -> None:
    """A pre-upgrade database kept one zone per shared config, so upgrading hands every run the last
    synced experiment's zone; re-syncing each experiment restores its own batches.

    Every planning path syncs first, so this lossy backfill never reaches a plan.
    """
    db_path = tmp_path / "experiments.sqlite"
    vmapped, per_lr = _grid("vmapped"), _grid("per-lr")
    vmapped.sync(db_path, static_names=lambda _: {"arch"})
    per_lr.sync(db_path, static_names=lambda _: {"arch", "lr"})
    _downgrade_to_config_zones(db_path, '{"dynamic_keys": [], "static_keys": ["arch", "lr"]}')
    with DatabaseManager(db_path) as database:
        database.initialize()
        assert len(_batch_shapes(database, "vmapped")) == 2

    vmapped.sync(db_path, static_names=lambda _: {"arch"})
    per_lr.sync(db_path, static_names=lambda _: {"arch", "lr"})

    with DatabaseManager(db_path) as database:
        assert _batch_shapes(database, "vmapped") == [({"arch": "mlp"}, 4)]
        assert _batch_shapes(database, "per-lr") == [({"arch": "mlp", "lr": 0.0003}, 2), ({"arch": "mlp", "lr": 0.001}, 2)]
