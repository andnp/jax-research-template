"""Private plotting helpers for reporting artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from research_plot.plots import plot_sensitivity


def _plot_hypers(
    target_hyperparameter: str,
    sorted_keys: list[object],
    slice_perf_sorted: Sequence[float | None],
    best_perf_sorted: list[float],
    experiment_slug: str,
    plot_suffix: str,
) -> Path:
    plot_path = Path("results/analysis") / experiment_slug / f"hyper_sensitivity{plot_suffix}.png"
    plot_sensitivity(target_hyperparameter, sorted_keys, slice_perf_sorted, best_perf_sorted, plot_path)
    return plot_path
