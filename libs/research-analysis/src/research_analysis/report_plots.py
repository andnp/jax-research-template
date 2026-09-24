"""Write report figures into an explicit output directory; computing a report never draws anything."""

from __future__ import annotations

import re
from pathlib import Path

from research_plot.plots import plot_distributions, plot_sensitivity

from research_analysis._reporting_models import BakeoffReport, HyperparameterSliceReport


def _file_stem(*parts: str) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9._-]+", "_", part) for part in parts if part)


def write_bakeoff_plots(report: BakeoffReport, output_dir: Path, name: str) -> dict[str, Path]:
    """Draw each block's per-run score distribution, returning block label -> figure path."""
    samples: dict[str, dict[str, list[float]]] = {block: {} for block in report.blocks}
    for run in report.runs:
        samples[run.block].setdefault(run.condition, []).append(run.value)

    paths: dict[str, Path] = {}
    for block, by_condition in samples.items():
        ordered = {condition: by_condition[condition] for condition in report.conditions if condition in by_condition}
        if not ordered:
            continue
        title = f"Per-run {report.metric_name}" + (f" ({block})" if block else "")
        path = output_dir / f"{_file_stem(name, 'run_distribution', block)}.png"
        paths[block] = plot_distributions(ordered, path, title=title, ylabel=report.metric_name)
    return paths


def write_slice_plot(report: HyperparameterSliceReport, output_dir: Path, name: str) -> Path:
    """Draw the best-achievable and champion-pinned sensitivity curves for ``report.vary``."""
    path = output_dir / f"{_file_stem(name, 'sensitivity', report.vary)}.png"
    return plot_sensitivity(report.vary, report.values, report.slice_performance, report.best_performance, path)
