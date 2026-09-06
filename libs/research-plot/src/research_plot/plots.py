"""Plotting functions that implement the monorepo science guide visualization rules.

Rules from science_guide.md:
- No top/right spines.
- No legend frames.
- Color consistency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True, slots=True)
class TrajectorySeries:
    """One trajectory with an optional pointwise uncertainty band."""

    x: Sequence[float]
    y: Sequence[float]
    lower: Sequence[float] | None = None
    upper: Sequence[float] | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryPanel:
    """Named trajectories sharing one y-axis."""

    ylabel: str
    series: Mapping[str, TrajectorySeries]


@dataclass(frozen=True, slots=True)
class EventWindow:
    """Time interval highlighted across trajectory panels."""

    start: float
    end: float
    label: str | None = None


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
    for i, pc in enumerate(cast(list[Any], parts["bodies"])):
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


def _validated_series(series: TrajectorySeries) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(series.x, dtype=np.float64)
    y = np.asarray(series.y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("trajectory x and y must be non-empty one-dimensional arrays of equal length.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("trajectory x and y must contain only finite values.")

    if (series.lower is None) != (series.upper is None):
        raise ValueError("trajectory lower and upper bands must be provided together.")
    if series.lower is not None and series.upper is not None:
        lower = np.asarray(series.lower, dtype=np.float64)
        upper = np.asarray(series.upper, dtype=np.float64)
        if lower.ndim != 1 or upper.ndim != 1 or len(lower) != len(y) or len(upper) != len(y):
            raise ValueError("trajectory uncertainty bands must match the y shape.")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("trajectory uncertainty bands must contain only finite values.")
        if np.any(lower > upper):
            raise ValueError("trajectory lower band cannot exceed its upper band.")
    return x, y


def plot_trajectory_panels(
    panels: Mapping[str, TrajectoryPanel],
    save_path: Path | str,
    *,
    title: str | None = None,
    x_label: str = "Step",
    event_windows: Sequence[EventWindow] = (),
) -> Path:
    """Render aligned trajectory panels with optional uncertainty bands and event windows."""
    if not panels:
        raise ValueError("panels must not be empty.")

    for window in event_windows:
        if not np.isfinite(window.start) or not np.isfinite(window.end) or window.start > window.end:
            raise ValueError("event windows must have finite start and end values with start <= end.")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13, max(3.0, 2.8 * len(panels))),
        sharex=True,
        squeeze=False,
    )

    for ax, (panel_name, panel) in zip(axes[:, 0], panels.items(), strict=True):
        _apply_minimal_style(ax)
        if not panel.series:
            raise ValueError(f"trajectory panel {panel_name!r} must contain at least one series.")

        reference_x: np.ndarray | None = None
        for label, series in panel.series.items():
            x, y = _validated_series(series)
            if reference_x is None:
                reference_x = x
            elif not np.array_equal(reference_x, x):
                raise ValueError(f"trajectory panel {panel_name!r} series must share the same x values.")

            line = ax.plot(x, y, label=label, linewidth=2.0)[0]
            if series.lower is not None and series.upper is not None:
                ax.fill_between(
                    x,
                    np.asarray(series.lower, dtype=np.float64),
                    np.asarray(series.upper, dtype=np.float64),
                    color=line.get_color(),
                    alpha=0.18,
                    linewidth=0.0,
                )

        for window in event_windows:
            ax.axvspan(window.start, window.end, color="#7f7f7f", alpha=0.12, linewidth=0.0)
            if window.label is not None:
                ax.text(
                    (window.start + window.end) / 2.0,
                    1.0,
                    window.label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
        ax.set_title(panel_name, loc="left")
        ax.set_ylabel(panel.ylabel)
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )

    axes[-1, 0].set_xlabel(x_label)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(
        rect=(0.0, 0.0, 0.74, 0.96) if title is not None else (0.0, 0.0, 0.74, 1.0)
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
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
