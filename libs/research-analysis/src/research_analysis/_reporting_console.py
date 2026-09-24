"""Private Rich presenters for reporting results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from research_analysis._reporting_models import (
    HyperparameterConfigReport,
    HyperparameterDifferenceReport,
    HyperparameterSliceReport,
)


def _present_best_hypers(
    report: HyperparameterConfigReport,
    experiment_slug: str,
    group_label: str | None,
) -> None:
    console = Console()
    console.print()
    subtitle = f"Experiment: {experiment_slug}" + (f" | Group: {group_label}" if group_label else "")
    console.print(
        Panel(
            Text("📊 HYPERPARAMETER ANALYSIS", style="bold cyan"),
            subtitle=subtitle,
        )
    )

    config_desc = ", ".join(f"{key}={value}" for key, value in report.winning_config.items())
    ci_low, ci_high = report.estimate_ci
    console.print(
        Panel(
            Text.assemble(
                ("Winning Configuration : ", "dim"), (f"{config_desc}\n"),
                ("Raw Winner Mean       : ", "dim"), (f"{report.raw_mean:.2f}\n"),
                ("Corrected Winner Mean : ", "bold green"), (f"{report.corrected_mean:.2f}\n"),
                ("Maximization Bias     : ", "bold red"), (f"+{report.maximization_bias:.2f}\n"),
                (f"Hierarchical Bootstrap {report.confidence_level:.0%} CI: ", "dim"),
                (f"[{ci_low:.2f}, {ci_high:.2f}]\n\n"),
                ("Pedagogical Interpretation:\n", "bold"),
                (
                    "Selecting the hyperparameter configuration with the highest raw average results "
                    "in Maximization Bias. This bias represents how much we have overestimated the algorithm's "
                    "true performance by selecting the 'lucky' configuration. The Corrected Mean resamples "
                    "these same seeds, re-running the selection each time, so it reflects which configuration "
                    "would be chosen as well as how it performs; it is not an estimate from held-out seeds."
                ),
            ),
            title="Maximization Bias Correction Log",
        )
    )


def _present_best_hypers_difference(report: HyperparameterDifferenceReport, experiment_slug: str) -> None:
    console = Console()
    console.print()
    console.print(
        Panel(
            Text("📊 HYPERPARAMETER DIFFERENCE", style="bold cyan"),
            subtitle=f"Experiment: {experiment_slug}",
        )
    )

    minuend_desc = ", ".join(f"{key}={value}" for key, value in report.minuend_config.items())
    subtrahend_desc = ", ".join(f"{key}={value}" for key, value in report.subtrahend_config.items())
    ci_low, ci_high = report.estimate_ci
    console.print(
        Panel(
            Text.assemble(
                ("Minuend Configuration    : ", "dim"), (f"{minuend_desc}\n"),
                ("Subtrahend Configuration : ", "dim"), (f"{subtrahend_desc}\n"),
                ("Mean Difference          : ", "bold green"), (f"{report.mean_diff:.2f}\n"),
                (f"Hierarchical Bootstrap {report.confidence_level:.0%} CI: ", "dim"),
                (f"[{ci_low:.2f}, {ci_high:.2f}]"),
            ),
            title="Paired Difference Report",
        )
    )


def _present_slice_hypers(
    report: HyperparameterSliceReport,
    experiment_slug: str,
    group_label: str | None,
) -> None:
    console = Console()
    console.print()
    subtitle = f"Experiment: {experiment_slug}" + (f" | Group: {group_label}" if group_label else "")
    console.print(
        Panel(
            Text(f"📊 HYPERPARAMETER SENSITIVITY: {report.vary}", style="bold cyan"),
            subtitle=subtitle,
        )
    )

    table = Table(title="Sensitivity Matrix", show_header=True, header_style="bold magenta")
    table.add_column("Value", style="dim")
    table.add_column("Best Achievable Mean", justify="right")
    table.add_column("Champion-Pinned Slice Mean", justify="right")
    for value, best, slice_val in zip(report.values, report.best_performance, report.slice_performance, strict=True):
        table.add_row(str(value), f"{best:.2f}", "—" if slice_val is None else f"{slice_val:.2f}")
    console.print(table)
