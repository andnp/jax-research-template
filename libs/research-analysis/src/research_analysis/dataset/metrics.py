"""Load and join per-run metric databases into typed frames."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

from research_analysis.dataset.types import MaterializedRun
from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode
from research_analysis.loader import load_sqlite_query


def _metric_query_pairs(pairs: list[tuple[int, int]], metrics: tuple[str, ...] | None) -> str:
    pair_sql = " OR ".join(f"(run_id = {run_id} AND execution_id = {execution_id})" for run_id, execution_id in pairs)
    metric_sql = ""
    if metrics is not None:
        quoted = ", ".join("'" + metric.replace("'", "''") + "'" for metric in metrics)
        metric_sql = f" AND metric_name IN ({quoted})"
    return (
        "SELECT run_id, execution_id, global_step, metric_name, value FROM metrics "
        f"WHERE ({pair_sql}){metric_sql} ORDER BY run_id, execution_id, metric_name, global_step"
    )


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
        frame = load_sqlite_query(metric_path, _metric_query_pairs(pairs, metrics))
        if frame.is_empty():
            for run_id, execution_id in pairs:
                diagnostics.add(DiagnosticCode.MISSING_METRIC_ARTIFACT, "metrics database has no requested rows", run_id, execution_id)
            continue
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
