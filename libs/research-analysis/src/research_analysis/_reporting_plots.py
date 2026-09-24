"""Private plotting helpers for reporting artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from research_plot.plots import plot_distributions, plot_sensitivity


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


def _plot_run_distributions(
    samples_by_block: dict[str, dict[str, list[float]]],
    metric_name: str,
    experiment_slug: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, samples in samples_by_block.items():
        suffix = f"__{label}" if label else ""
        plot_path = Path("results/analysis") / experiment_slug / f"run_distribution{suffix}.png"
        title = f"Per-run {metric_name}" + (f" ({label})" if label else "")
        paths[label] = plot_distributions(samples, plot_path, title=title, ylabel=metric_name)
    return paths
