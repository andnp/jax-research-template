import sqlite3
from pathlib import Path

import polars as pl
from research_analysis.dataset.metrics import load_scalar_metric_frames
from research_instrument.scalar import write_execution_scalars, write_seed_batch_scalars


def test_load_scalar_metric_frames_reads_selected_scopes_and_identities(tmp_path: Path) -> None:
    """
    Load selected run and execution scalars while preserving their stored identities.
    """
    database_path = tmp_path / "metrics.sqlite"
    write_seed_batch_scalars(
        database_path,
        {"score": (1.5, float("nan")), "ignored": (3.0, 4.0)},
        experiment_id=10,
        execution_id=100,
        run_ids=(21, 22),
        seed_ids=(0, 1),
    )
    write_execution_scalars(
        database_path,
        {"aggregate": 7.5},
        experiment_id=10,
        execution_id=100,
    )
    write_seed_batch_scalars(
        database_path,
        {"score": (99.0,)},
        experiment_id=11,
        execution_id=100,
        run_ids=(21,),
        seed_ids=(0,),
    )

    frames = load_scalar_metric_frames(
        database_path,
        experiment_id=10,
        run_execution_pairs=[(21, 100), (22, 100)],
        execution_ids=[100, 100],
        metrics=("score", "aggregate"),
    )
    rows = (
        pl.concat(frames)
        .select("scope", "metric_name", "experiment_id", "run_id", "seed_id", "execution_id", "value")
        .sort("scope", "metric_name", "run_id")
        .rows()
    )

    assert rows == [
        ("execution", "aggregate", 10, None, None, 100, 7.5),
        ("run", "score", 10, 21, 0, 100, 1.5),
        ("run", "score", 10, 22, 1, 100, None),
    ]


def test_load_scalar_metric_frames_accepts_curve_only_databases(tmp_path: Path) -> None:
    """
    Read older curve databases that do not have the additive scalar table.
    """
    database_path = tmp_path / "metrics.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE metrics (run_id INTEGER, execution_id INTEGER, metric_name TEXT, global_step INTEGER, value REAL)"
        )

    frames = load_scalar_metric_frames(
        database_path,
        experiment_id=10,
        run_execution_pairs=[(21, 100)],
        execution_ids=[100],
        metrics=None,
    )

    assert frames == []
