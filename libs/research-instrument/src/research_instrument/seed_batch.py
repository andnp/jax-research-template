"""Persist a seed-batched metric curve, one row per run.

A vmapped training run produces a seed-major curve array of shape
``(n_seeds, n_steps)``: the runner hands ``train_fn`` a batch of runs that
share a hyperparameter cell but differ by seed, so row ``i`` of the array
belongs to run ``i`` and must be written under that run's own ``run_id`` — a
silent seed/run mispairing would attribute one seed's results to a different
run with no error. Curves arrive already thinned, one value kept in every
``every`` steps, and are stored at the steps they were taken from.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from research_instrument.sqlite_backend import INSERT_METRICS_SQL, open_metrics_connection


def write_seed_batch_curve(
    database_path: Path | str,
    curves: Sequence[Sequence[float]],
    *,
    metric_name: str,
    experiment_id: int,
    execution_id: int,
    run_ids: Sequence[int],
    seed_ids: Sequence[int],
    every: int = 1,
    batch_size: int = 512,
) -> None:
    """Write each row of a seed-major curve array under its own run id.

    ``curves``, ``run_ids``, and ``seed_ids`` are zipped positionally: row
    ``i`` of ``curves`` is written with ``run_id=run_ids[i]`` and
    ``seed_id=seed_ids[i]``. All three must have the same length — callers
    are responsible for keeping them in the same row order, since this is
    exactly the seed/run pairing that must not be silently scrambled.

    Args:
        database_path: Path to the metrics SQLite database. Parent
            directories are created if absent.
        curves: Seed-major metric values, one row (sequence of floats) per
            run, e.g. the ``(n_seeds, n_steps)`` output of a vmapped
            training run.
        metric_name: Metric name recorded for every frame written by this
            call.
        experiment_id: Control-plane experiment id shared by every row.
        execution_id: Control-plane execution id shared by every row.
        run_ids: Run id for each row of ``curves``, same length and order.
        seed_ids: Seed id for each row of ``curves``, same length and order.
        every: Row value ``j`` was taken at step ``j * every``, and is stored
            at that step.
        batch_size: Rows are inserted in chunks of this size within the
            single transaction.
    """
    resolved_path = Path(database_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if every < 1:
        raise ValueError(f"every must be >= 1; received {every}.")

    rows_to_insert: list[tuple[str, float, int, int, int, int, int]] = []
    for row, run_id, seed_id in zip(curves, run_ids, seed_ids, strict=True):
        rows_to_insert.extend(
            (metric_name, float(value), index * every, seed_id, experiment_id, run_id, execution_id)
            for index, value in enumerate(row)
        )

    conn = open_metrics_connection(resolved_path)
    try:
        for batch_start in range(0, len(rows_to_insert), batch_size):
            conn.executemany(
                INSERT_METRICS_SQL,
                rows_to_insert[batch_start : batch_start + batch_size],
            )
        conn.commit()
    finally:
        conn.close()
