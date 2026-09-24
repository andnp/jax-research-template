"""Private Rich presenters for reporting results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from research_analysis._reporting_models import (
    BakeoffReport,
    CellSummary,
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

    console.print(
        Panel(
            Text(f"Saved sensitivity visual artifact to: {report.plot_path}", style="italic green"),
            title="Artifact Outputs Panel",
        )
    )


def _format_cell(cell: CellSummary) -> str:
    if cell.estimate is None:
        return "— (n=0)"
    interval = "" if cell.ci is None else f" [{cell.ci[0]:.3g}, {cell.ci[1]:.3g}]"
    spread = "" if cell.sd is None else f" SD {cell.sd:.2g}"
    tolerance = (
        f" TI [{cell.tolerance[0]:.3g}, {cell.tolerance[1]:.3g}]"
        if cell.tolerance is not None
        else f" TI needs n≥{cell.tolerance_min_runs}"
    )
    return f"{cell.estimate:.3g}{interval}{spread}{tolerance} (n={cell.n}, median run {cell.median_run_id})"


def _present_bakeoff_report(report: BakeoffReport, experiment_slug: str) -> None:
    console = Console()
    console.print()
    console.print(Panel(Text(f"🏆 BAKEOFF REPORT: {report.conditions}", style="bold cyan"), subtitle=f"Experiment: {experiment_slug}"))

    blocks = sorted({label for per_condition in report.cells.values() for label in per_condition})
    title = (
        f"Cells ({report.metric_name}: {report.spec.estimand} [{report.spec.confidence_level:.0%} CI], SD, "
        f"({report.spec.tolerance_confidence:.0%}, {report.spec.tolerance_coverage:.0%}) tolerance interval)"
    )
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Condition")
    for label in blocks:
        table.add_column(label or "(no blocking)", justify="right")
    for condition in report.conditions:
        table.add_row(condition, *[_format_cell(report.cells[condition][label]) for label in blocks])
    console.print(table)
    figures = "\n".join(f"{label or '(no blocking)'}: {path}" for label, path in report.distribution_plot_paths.items())
    console.print(Panel(Text(figures, style="italic green"), title="Per-run distribution figures"))

    if report.champions is not None:
        champ_table = Table(title="Champion configurations", show_header=True, header_style="bold magenta")
        champ_table.add_column("Block")
        champ_table.add_column("Condition")
        champ_table.add_column("Config")
        champ_table.add_column("Corrected mean")
        for label, per_condition in report.champions.items():
            for condition, champion in per_condition.items():
                config_desc = ", ".join(f"{key}={value}" for key, value in champion.winning_config.items())
                champ_table.add_row(label or "(no blocking)", condition, config_desc, f"{champion.corrected_mean:.3g}")
        console.print(champ_table)

    if report.warnings:
        lines = [f"• [{w.code.value}] {w.condition or '(all)'} {dict(w.cell) or ''}: {w.message}" for w in report.warnings]
        console.print(Panel(Text("\n".join(lines)), title=f"⚠ Warnings ({len(report.warnings)})", style="yellow"))
    else:
        console.print(Panel(Text("No warnings."), style="green"))

    confidences = sorted({item.confidence for item in report.differences})
    title = (
        f"Paired differences by block ({report.family_size} comparisons, Bonferroni-adjusted to "
        f"{report.spec.confidence_level:.0%} jointly)"
    )
    diff_table = Table(title=title, show_header=True, header_style="bold magenta")
    diff_table.add_column("Block")
    diff_table.add_column("A")
    diff_table.add_column("B")
    diff_table.add_column("A − B")
    diff_table.add_column(" / ".join(f"{confidence * 100:.4g}%" for confidence in confidences) + " CI")
    diff_table.add_column("n")
    diff_table.add_column("Holm p")
    diff_table.add_column("Method / note")
    for item in report.differences:
        if item.estimate is None or item.ci is None:
            diff_table.add_row(item.block or "(no blocking)", item.minuend, item.subtrahend, "—", "—", str(item.n_paired), "—", f"skipped: {item.skipped_reason}")
            continue
        p_value = "—" if item.p_value is None else f"{item.p_value:.4f}"
        interval = f"[{item.ci[0]:+.3g}, {item.ci[1]:+.3g}]"
        diff_table.add_row(item.block or "(no blocking)", item.minuend, item.subtrahend, f"{item.estimate:+.3g}", interval, str(item.n_paired), p_value, item.method)
    console.print(diff_table)
