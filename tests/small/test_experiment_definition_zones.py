"""Contract tests for vmap-zone batching when experiments share a database."""

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
    _grid("vmapped").sync(db_path, static_parameter_names=frozenset({"arch"}))
    _grid("per-lr").sync(db_path, static_parameter_names=frozenset({"arch", "lr"}))

    with DatabaseManager(db_path) as database:
        vmapped = _batch_shapes(database, "vmapped")
        per_lr = _batch_shapes(database, "per-lr")

    assert vmapped == [({"arch": "mlp"}, 4)]
    assert per_lr == [({"arch": "mlp", "lr": 0.0003}, 2), ({"arch": "mlp", "lr": 0.001}, 2)]
