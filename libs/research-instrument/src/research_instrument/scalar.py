"""Persistence helpers for scalar experiment metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from research_instrument.sqlite_backend import open_metrics_connection

type _ScalarMetricRow = tuple[str, float, str, int, int | None, int | None, int]


def write_seed_batch_scalars(
    database_path: Path | str,
    metrics: Mapping[str, Sequence[float]],
    *,
    experiment_id: int,
    execution_id: int,
    run_ids: Sequence[int],
    seed_ids: Sequence[int],
) -> None:
    """Write one scalar value per run for each metric in a seed batch."""
    if len(run_ids) != len(seed_ids):
        raise ValueError("run_ids and seed_ids must have the same length")

    rows: list[_ScalarMetricRow] = []
    for metric_name, values in metrics.items():
        if len(values) != len(run_ids):
            raise ValueError(f"metric {metric_name!r} must have one value per run")
        rows.extend(
            (metric_name, float(value), "run", experiment_id, run_id, seed_id, execution_id)
            for value, run_id, seed_id in zip(values, run_ids, seed_ids, strict=True)
        )

    _write_scalar_rows(database_path, rows)


def write_execution_scalars(
    database_path: Path | str,
    metrics: Mapping[str, float],
    *,
    experiment_id: int,
    execution_id: int,
) -> None:
    """Write scalar values shared by every run in an execution."""
    rows: list[_ScalarMetricRow] = [
        (metric_name, float(value), "execution", experiment_id, None, None, execution_id)
        for metric_name, value in metrics.items()
    ]
    _write_scalar_rows(database_path, rows)


def _write_scalar_rows(database_path: Path | str, rows: list[_ScalarMetricRow]) -> None:
    if not rows:
        return

    resolved_path = Path(database_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_metrics_connection(resolved_path)
    try:
        connection.executemany(
            "INSERT INTO scalar_metrics ("
            "metric_name, value, scope, experiment_id, run_id, seed_id, execution_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
