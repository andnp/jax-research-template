"""Private plotting helpers for reporting artifacts."""

from __future__ import annotations

from pathlib import Path

from research_plot.plots import plot_ecdf, plot_sensitivity


def _plot_hypers(
    target_hyperparameter: str,
    sorted_keys: list[object],
    slice_perf_sorted: list[float],
    best_perf_sorted: list[float],
    experiment_slug: str,
    plot_suffix: str,
) -> Path:
    plot_path = Path("results/analysis") / experiment_slug / f"hyper_sensitivity{plot_suffix}.png"
    plot_sensitivity(target_hyperparameter, sorted_keys, slice_perf_sorted, best_perf_sorted, plot_path)
    return plot_path


def _plot_bakeoff(
    score_curves: dict[str, tuple[list[float], list[float]]],
    experiment_slug: str,
) -> Path | None:
    plot_path = Path("results/analysis") / experiment_slug / "normalized_score_comparison.png"
    if score_curves:
        plot_ecdf(score_curves, plot_path)
        return plot_path
    return None
