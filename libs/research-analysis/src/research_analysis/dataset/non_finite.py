"""Surface diverged (NaN or infinite) metric values as per-run outcomes."""

from __future__ import annotations

import polars as pl

from research_analysis.diagnostics.accumulator import DiagnosticAccumulator
from research_analysis.diagnostics.types import DiagnosticCode


def non_finite_summary(metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        metrics.filter(~pl.col("value").is_finite())
        .group_by(["run_id", "execution_id", "seed", "metric"])
        .agg(pl.col("step").min().alias("first_step"), pl.len().alias("count"))
        .sort(["run_id", "execution_id", "metric"])
    )


def diagnose_non_finite(summary: pl.DataFrame, diagnostics: DiagnosticAccumulator) -> None:
    for row in summary.iter_rows(named=True):
        diagnostics.add(
            DiagnosticCode.NON_FINITE_METRIC,
            f"run {row['run_id']} (seed {row['seed']}) metric {row['metric']!r} is non-finite from step "
            f"{row['first_step']} ({row['count']} values)",
            row["run_id"],
            row["execution_id"],
        )
