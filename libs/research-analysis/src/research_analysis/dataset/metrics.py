"""Load and join per-run metric databases into typed frames."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import polars as pl

from research_analysis.dataset.types import MaterializedRun
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode
from research_analysis.loader import load_sqlite_query

_MAX_SQL_PARAMETERS = 900


def _metric_query_pairs(
    pairs: list[tuple[int, int]], metrics: tuple[str, ...] | None
) -> tuple[str, tuple[object, ...]]:
    pair_sql = " OR ".join("(run_id = ? AND execution_id = ?)" for _ in pairs)
    parameters: list[object] = [value for pair in pairs for value in pair]
    metric_sql = ""
    if metrics is not None:
        placeholders = ", ".join("?" for _ in metrics)
        metric_sql = f" AND metric_name IN ({placeholders})"
        parameters.extend(metrics)
    query = (
        "SELECT run_id, execution_id, global_step, metric_name, value FROM metrics "
        f"WHERE ({pair_sql}){metric_sql} ORDER BY run_id, execution_id, metric_name, global_step"
    )
    return query, tuple(parameters)


def _metric_query_batches(
    pairs: list[tuple[int, int]], metrics: tuple[str, ...] | None
) -> list[tuple[str, tuple[object, ...]]]:
    metric_batches = [metrics] if metrics is None else [
        metrics[start : start + _MAX_SQL_PARAMETERS - 2] for start in range(0, len(metrics), _MAX_SQL_PARAMETERS - 2)
    ]
    queries: list[tuple[str, tuple[object, ...]]] = []
    for metric_batch in metric_batches:
        pair_limit = max(1, (_MAX_SQL_PARAMETERS - (len(metric_batch) if metric_batch is not None else 0)) // 2)
        for start in range(0, len(pairs), pair_limit):
            queries.append(_metric_query_pairs(pairs[start : start + pair_limit], metric_batch))
    return queries


def load_metric_frames(
    rows: list[MaterializedRun], metrics: tuple[str, ...] | None, diagnostics: DiagnosticAccumulator
) -> list[pl.DataFrame]:
    metric_groups: defaultdict[Path, list[MaterializedRun]] = defaultdict(list)
    for row in rows:
        metric_groups[row.metrics_db_path].append(row)

    metric_frames: list[pl.DataFrame] = []
    for metric_path, metric_rows in metric_groups.items():
        pairs = [(row.run_id, row.execution_id) for row in metric_rows]
        if not metric_path.exists():
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, f"metrics database does not exist: {metric_path}", run_id, execution_id)
            continue
        try:
            frames = [
                load_sqlite_query(metric_path, query, parameters)
                for query, parameters in _metric_query_batches(pairs, metrics)
            ]
        except sqlite3.DatabaseError as error:
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MALFORMED_ARTIFACT, f"metrics database {metric_path} is unreadable: {error}", run_id, execution_id)
            continue
        frames = [frame for frame in frames if not frame.is_empty()]
        if not frames:
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "metrics database has no requested rows", run_id, execution_id)
            continue
        # The metric store records NaN as NULL; restore it so divergence is never skipped as missing.
        frame = pl.concat(frames).with_columns(pl.col("value").cast(pl.Float64).fill_null(float("nan")))
        present = set(frame.select(["run_id", "execution_id"]).unique().iter_rows())
        for run_id, execution_id in pairs:
            if (run_id, execution_id) not in present:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, f"no requested metric rows in {metric_path}", run_id, execution_id)
        metadata = pl.DataFrame(
            [
                {
                    **row.dimension_values,
                    "seed": row.seed,
                    "run_id": row.run_id,
                    "execution_id": row.execution_id,
                    "ablation": row.ablation,
                }
                for row in metric_rows
            ]
        )
        metric_frames.append(
            frame.join(metadata, on=["run_id", "execution_id"], how="inner")
            .rename({"global_step": "step", "metric_name": "metric"})
            .select(
                [
                    "condition_name",
                    *(["env_name"] if "env_name" in metadata.columns else []),
                    "seed",
                    "run_id",
                    "execution_id",
                    "ablation",
                    "step",
                    "metric",
                    "value",
                ]
            )
        )
    return metric_frames
