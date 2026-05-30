"""Plotting functions that implement the monorepo science guide visualization rules.

Rules from science_guide.md:
- No top/right spines.
- No legend frames.
- Color consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _apply_minimal_style(ax: plt.Axes) -> None:
    """Removes top/right spines and sets neat labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_distributions(
    data_a: np.ndarray,
    data_b: np.ndarray,
    label_a: str,
    label_b: str,
    save_path: Path | str,
    title: str = "Performance Distributions",
) -> Path:
    """Generate a clean violin plot comparing two performance distributions.

    Args:
        data_a: 1-D array of outcomes for Method A.
        data_b: 1-D array of outcomes for Method B.
        label_a: Name of Method A.
        label_b: Name of Method B.
        save_path: Location to save the plot.
        title: Chart title.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_minimal_style(ax)

    # Convert to standard float lists/arrays for matplotlib
    vals = [np.asarray(data_a, dtype=np.float64), np.asarray(data_b, dtype=np.float64)]
    
    parts = ax.violinplot(vals, showmeans=True, showmedians=True, showextrema=False)
    
    # Custom styling for violin bodies
    colors = ["#1f77b4", "#ff7f0e"]  # Consistent Blue & Orange
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
        
    ax.set_xticks([1, 2])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Outcome Metric")
    ax.set_title(title, pad=15)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_sensitivity(
    param_name: str,
    values: list[Any],
    slice_performance: list[float],
    best_performance: list[float],
    save_path: Path | str,
) -> Path:
    """Generate a sensitivity plot comparing the 'Best' and 'Slice' configurations.

    Args:
        param_name: Name of target hyperparameter.
        values: Sweep values.
        slice_performance: Performance along the optimal slice (fixed remaining hypers).
        best_performance: Maximum achievable performance at each value of target hyperparameter.
        save_path: Location to save the plot.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    _apply_minimal_style(ax)

    x = np.arange(len(values))
    ax.plot(x, best_performance, label="Best Achievable", color="#2ca02c", marker="o", linewidth=2)
    ax.plot(x, slice_performance, label="Optimal Slice", color="#d62728", marker="s", linestyle="--", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(param_name)
    ax.set_ylabel("Performance")
    ax.set_title(f"Sensitivity: {param_name}", pad=15)

    # Minimalist legend: no frame
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_ecdf(
    ecdf_curves: dict[str, tuple[list[float], list[float]]],
    save_path: Path | str,
) -> Path:
    """Generate ECDF curves comparing multiple methods.

    Args:
        ecdf_curves: Maps method label to (sorted_scores, ecdf_proportions).
        save_path: Location to save the plot.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_minimal_style(ax)

    for label, (scores, proportions) in ecdf_curves.items():
        ax.step(scores, proportions, label=label, where="post", linewidth=2)

    ax.set_xlabel("Normalized Score (ECDF)")
    ax.set_ylabel("Fraction of Task-Seeds")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Empirical Cumulative Distribution Function (ECDF)", pad=15)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
