"""Automated statistical reporting API for RL experiments.

Implements pairwise comparisons, hyperparameter sensitivity analysis, and benchmark bakeoffs
with pedagogical explanations and diagnostic plotting.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import polars as pl
import scipy.stats as stats
from experiment_definition.db import DatabaseManager
from research_plot import plot_distributions, plot_ecdf, plot_sensitivity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from research_analysis.experiment import load_experiment_metrics
from research_analysis.hypothesis import mann_whitney_u_test, welch_ttest


@dataclass(frozen=True)
class StatisticalTestDetails:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    is_significant: bool
    assumptions: dict[str, Any]
    justification: str
    intermediate_tests: dict[str, StatisticalTestDetails]


@dataclass(frozen=True)
class ABComparisonReport:
    experiment_name: str
    condition_a: str
    condition_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    difference_in_means: float
    difference_ci: tuple[float, float]
    test_details: StatisticalTestDetails
    distribution_plot_path: Path | None


@dataclass(frozen=True)
class HyperparameterSensitivityReport:
    target_hyperparameter: str
    winning_value: Any
    raw_mean: float
    corrected_mean: float
    maximization_bias: float
    sensitivity_slice: dict[Any, float]
    sensitivity_best: dict[Any, float]
    sensitivity_plot_path: Path | None


@dataclass(frozen=True)
class BenchmarkBakeoffReport:
    algorithms: list[str]
    environments: list[str]
    ecdf_scores: dict[str, dict[str, float]]
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float]]
    ecdf_plot_path: Path | None


@dataclass(frozen=True)
class _PairwiseStatistics:
    test_details: StatisticalTestDetails
    difference_ci: tuple[float, float]
    normality_p_values: tuple[float, float]
    normality_flags: tuple[bool, bool]


# ── Database Helpers ─────────────────────────────────────────────────────────


def _resolve_metrics_db_path(root_path: str) -> Path:
    execution_root = Path(root_path)
    for candidate in (execution_root, *execution_root.parents):
        db = candidate / "metrics.sqlite"
        if db.exists():
            return db
    return execution_root.parent / "metrics.sqlite"


def _load_run_metric(db_path: Path, run_id: int, metric_name: str) -> np.ndarray | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT global_step, value FROM metrics WHERE run_id = ? AND metric_name = ? ORDER BY global_step",
            (run_id, metric_name),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return None
        return np.array([float(r[1]) for r in rows], dtype=np.float64)
    except Exception:
        return None


# ── Entry Points ─────────────────────────────────────────────────────────────


def _load_pairwise_data(
    db_path: Path,
    experiment_slug: str,
    condition_a: str,
    condition_b: str,
    metric_name: str,
) -> tuple[str, np.ndarray, np.ndarray, bool]:
    with DatabaseManager(db_path) as database:
        database.initialize()
        exp_row = database.get_experiment(experiment_slug)
        if exp_row is None:
            raise ValueError(f"Unknown experiment {experiment_slug!r}")

    all_metrics = load_experiment_metrics(
        experiments_db=db_path,
        slug=experiment_slug,
        metrics=[metric_name],
    )
    final_per_run = (
        all_metrics
        .group_by(["condition_name", "seed"])
        .agg(pl.col("value").last())
    )
    rows_a = final_per_run.filter(pl.col("condition_name") == condition_a).sort("seed")
    rows_b = final_per_run.filter(pl.col("condition_name") == condition_b).sort("seed")
    if rows_a.is_empty() or rows_b.is_empty():
        raise ValueError(
            f"No completed runs found for condition_name={condition_a!r} and condition_name={condition_b!r}"
        )

    seeds_a = rows_a["seed"].to_list()
    seeds_b = rows_b["seed"].to_list()
    return exp_row.name, rows_a["value"].to_numpy(), rows_b["value"].to_numpy(), seeds_a == seeds_b


def _compute_pairwise_statistics(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    *,
    alpha: float,
    is_paired: bool,
) -> _PairwiseStatistics:
    shapiro_a = stats.shapiro(vals_a) if len(vals_a) >= 3 else (1.0, 1.0)
    shapiro_b = stats.shapiro(vals_b) if len(vals_b) >= 3 else (1.0, 1.0)
    normal_a = shapiro_a[1] > 0.05
    normal_b = shapiro_b[1] > 0.05

    norm_a_details = StatisticalTestDetails(
        test_name="Shapiro-Wilk (Group A)",
        statistic=shapiro_a[0],
        p_value=shapiro_a[1],
        effect_size=None,
        is_significant=not normal_a,
        assumptions={},
        justification="Checks if Group A outcomes deviate significantly from a normal distribution.",
        intermediate_tests={},
    )
    norm_b_details = StatisticalTestDetails(
        test_name="Shapiro-Wilk (Group B)",
        statistic=shapiro_b[0],
        p_value=shapiro_b[1],
        effect_size=None,
        is_significant=not normal_b,
        assumptions={},
        justification="Checks if Group B outcomes deviate significantly from a normal distribution.",
        intermediate_tests={},
    )
    intermediate_tests = {"normality_a": norm_a_details, "normality_b": norm_b_details}

    if is_paired:
        differences = vals_a - vals_b
        shapiro_diff = stats.shapiro(differences) if len(differences) >= 3 else (1.0, 1.0)
        normal_diff = shapiro_diff[1] > 0.05
        intermediate_tests["normality_diff"] = StatisticalTestDetails(
            test_name="Shapiro-Wilk (Differences)",
            statistic=shapiro_diff[0],
            p_value=shapiro_diff[1],
            effect_size=None,
            is_significant=not normal_diff,
            assumptions={},
            justification="Checks if paired differences are normally distributed.",
            intermediate_tests={},
        )

        if normal_diff:
            ttest = stats.ttest_rel(vals_a, vals_b)
            p_value = float(ttest.pvalue)
            statistic = float(ttest.statistic)
            test_name = "Paired t-test"
            justification = (
                "Selected because the same random seeds were reused (paired data) "
                "and differences are normally distributed (Shapiro-Wilk p > 0.05)."
            )
        else:
            wilc = stats.wilcoxon(vals_a, vals_b)
            p_value = float(wilc.pvalue)
            statistic = float(wilc.statistic)
            test_name = "Wilcoxon Signed-Rank Test"
            justification = (
                "Selected because the same random seeds were reused (paired data) "
                "but the differences deviate significantly from normality (Shapiro-Wilk p <= 0.05)."
            )
        effect_size = float(np.mean(differences) / np.std(differences, ddof=1)) if np.std(differences, ddof=1) > 0 else 0.0
    elif normal_a and normal_b:
        wel = welch_ttest(vals_a, vals_b)
        p_value = wel.p_value
        statistic = wel.t_statistic
        test_name = "Welch's t-test"
        justification = (
            "Selected because the groups are independent (unpaired seeds) "
            "and both are normally distributed."
        )
        effect_size = (np.mean(vals_a) - np.mean(vals_b)) / math.sqrt((np.var(vals_a, ddof=1) + np.var(vals_b, ddof=1)) / 2)
    else:
        mw = mann_whitney_u_test(vals_a, vals_b)
        p_value = mw.p_value
        statistic = mw.u_statistic
        test_name = "Mann-Whitney U-test"
        justification = (
            "Selected because the groups are independent (unpaired seeds) "
            "and at least one group deviates from normality."
        )
        effect_size = mw.rank_biserial_correlation

    is_significant = p_value < alpha
    rng = np.random.default_rng(12345)
    boot_diffs = []
    n_a = len(vals_a)
    n_b = len(vals_b)
    for _ in range(1000):
        if is_paired:
            indices = rng.choice(n_a, size=n_a, replace=True)
            boot_diffs.append(np.mean(vals_a[indices] - vals_b[indices]))
        else:
            idx_a = rng.choice(n_a, size=n_a, replace=True)
            idx_b = rng.choice(n_b, size=n_b, replace=True)
            boot_diffs.append(np.mean(vals_a[idx_a]) - np.mean(vals_b[idx_b]))

    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return _PairwiseStatistics(
        test_details=StatisticalTestDetails(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            assumptions={"paired": is_paired, "normal_a": normal_a, "normal_b": normal_b},
            justification=justification,
            intermediate_tests=intermediate_tests,
        ),
        difference_ci=(float(ci_low), float(ci_high)),
        normality_p_values=(float(shapiro_a[1]), float(shapiro_b[1])),
        normality_flags=(normal_a, normal_b),
    )


def _load_hyperparameter_records(
    db_path: Path,
    experiment_slug: str,
    target_hyperparameter: str,
    metric_name: str,
) -> list[dict[str, Any]]:
    with DatabaseManager(db_path) as database:
        database.initialize()
        exp_row = database.get_experiment(experiment_slug)
        if exp_row is None:
            raise ValueError(f"Unknown experiment {experiment_slug!r}")
        runs = database.list_runs(exp_row.id)

    records: list[dict[str, Any]] = []
    with DatabaseManager(db_path) as database:
        for run in runs:
            latest_exec = database.get_latest_completed_execution_for_run(run.id)
            latest_art = database.get_latest_completed_artifacts_for_run(run.id)
            if latest_exec is None or latest_art is None:
                continue

            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                continue
            hypers = json.loads(hyper_config.json_blob)
            val = hypers.get(target_hyperparameter)
            if val is None:
                continue

            metrics_db = _resolve_metrics_db_path(latest_art.root_path)
            metric_curve = _load_run_metric(metrics_db, run.id, metric_name)
            if metric_curve is None:
                continue
            records.append({"run_id": run.id, "value": float(metric_curve[-1]), "hypers": hypers})

    if not records:
        raise ValueError(f"No completed runs found containing hyperparameter {target_hyperparameter!r}")
    return records


def compare_pairwise(
    db_path: Path | str,
    experiment_slug: str,
    condition_a: str,
    condition_b: str,
    metric_name: str,
    confidence_level: float = 0.95,
    verbose: bool = True,
) -> ABComparisonReport:
    """Compare exactly two experimental conditions, selecting the correct test automatically."""
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    alpha = 1.0 - confidence_level
    db_path = Path(db_path)
    experiment_name, vals_a, vals_b, is_paired = _load_pairwise_data(
        db_path,
        experiment_slug,
        condition_a,
        condition_b,
        metric_name,
    )

    pairwise_statistics = _compute_pairwise_statistics(
        vals_a,
        vals_b,
        alpha=alpha,
        is_paired=is_paired,
    )
    test_details = pairwise_statistics.test_details
    ci_low, ci_high = pairwise_statistics.difference_ci
    shapiro_a_p, shapiro_b_p = pairwise_statistics.normality_p_values
    normal_a, normal_b = pairwise_statistics.normality_flags

    # 4. Distribution Plot
    analysis_dir = Path("results/analysis") / experiment_slug
    plot_path = analysis_dir / "ab_comparison.png"
    plot_distributions(vals_a, vals_b, condition_a, condition_b, plot_path)

    report = ABComparisonReport(
        experiment_name=experiment_name,
        condition_a=condition_a,
        condition_b=condition_b,
        metric_name=metric_name,
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        difference_in_means=float(np.mean(vals_a) - np.mean(vals_b)),
        difference_ci=(float(ci_low), float(ci_high)),
        test_details=test_details,
        distribution_plot_path=plot_path,
    )

    if verbose:
        console = Console()
        console.print()
        console.print(
            Panel(
                Text(f"🔬 PAIRWISE REPORT: {report.experiment_name}", style="bold cyan"),
                subtitle=f"Metric: {metric_name}",
            )
        )

        table = Table(title="Descriptive Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Group", style="dim")
        table.add_column("N", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_row(condition_a, str(len(vals_a)), f"{report.mean_a:.2f}", f"{np.std(vals_a, ddof=1):.2f}")
        table.add_row(condition_b, str(len(vals_b)), f"{report.mean_b:.2f}", f"{np.std(vals_b, ddof=1):.2f}")
        console.print(table)

        just_panel = Panel(
            Text.assemble(
                ("Hypothesis Test Selection:\n", "bold"),
                (f"• Selected Test: {test_details.test_name}\n", "yellow"),
                (f"• Justification: {test_details.justification}\n\n", "italic"),
                ("Assumptions Checked:\n", "bold"),
                (f"  - Repeated measures (paired seeds): {'Yes' if is_paired else 'No'}\n"),
                (f"  - Group A normal distribution (p={shapiro_a_p:.4f}): {'Yes' if normal_a else 'No'}\n"),
                (f"  - Group B normal distribution (p={shapiro_b_p:.4f}): {'Yes' if normal_b else 'No'}"),
            ),
            title="Methodology Justification Box",
        )
        console.print(just_panel)

        sig_style = "bold green" if test_details.is_significant else "bold red"
        sig_text = "Statistically Significant" if test_details.is_significant else "Not Statistically Significant"
        findings = Panel(
            Text.assemble(
                ("Statistic       : ", "dim"), (f"{test_details.statistic:.4f}\n"),
                ("p-value         : ", "dim"), (f"{test_details.p_value:.4f} ", sig_style), (f"({sig_text})\n", sig_style),
                ("Difference Mean : ", "dim"), (f"{report.difference_in_means:.4f}\n"),
                ("Bootstrapped CI : ", "dim"), (f"[{report.difference_ci[0]:.2f}, {report.difference_ci[1]:.2f}]\n\n"),
                ("Pedagogical Interpretation:\n", "bold"),
                (
                    f"There is a {report.test_details.p_value * 100:.2f}% chance that we would observe "
                    "a difference this large purely due to random variations if the two algorithms "
                    "were completely identical in performance."
                ),
            ),
            title="Findings & Interpretation",
        )
        console.print(findings)

        console.print(
            Panel(
                Text(f"Saved distribution visual artifact to: {plot_path}", style="italic green"),
                title="Artifact Outputs Panel",
            )
        )

    return report


@overload
def analyze_hypers(
    db_path: Path | str,
    experiment_slug: str,
    target_hyperparameter: str,
    metric_name: str,
    group_by: None = None,
    verbose: bool = True,
) -> HyperparameterSensitivityReport: ...


@overload
def analyze_hypers(
    db_path: Path | str,
    experiment_slug: str,
    target_hyperparameter: str,
    metric_name: str,
    group_by: list[str],
    verbose: bool = True,
) -> dict[tuple[Any, ...], HyperparameterSensitivityReport]: ...


def analyze_hypers(
    db_path: Path | str,
    experiment_slug: str,
    target_hyperparameter: str,
    metric_name: str,
    group_by: list[str] | None = None,
    verbose: bool = True,
) -> HyperparameterSensitivityReport | dict[tuple[Any, ...], HyperparameterSensitivityReport]:
    """Analyze hyperparameter sweep, correcting for maximization bias via two-stage bootstrapping.

    When ``group_by`` names one or more hyperparameter keys, the analysis is performed
    separately for each distinct combination of those keys' values (read from each run's
    stored hyperparameter JSON), and a mapping from the group key tuple to its own
    ``HyperparameterSensitivityReport`` is returned instead of a single pooled report.
    """
    db_path = Path(db_path)
    records = _load_hyperparameter_records(
        db_path,
        experiment_slug,
        target_hyperparameter,
        metric_name,
    )

    if group_by:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in records:
            group_key = tuple(record["hypers"].get(key) for key in group_by)
            groups.setdefault(group_key, []).append(record)

        return {
            group_key: _analyze_hypers_group(
                group_records,
                target_hyperparameter=target_hyperparameter,
                experiment_slug=experiment_slug,
                verbose=verbose,
                group_label=", ".join(f"{key}={value}" for key, value in zip(group_by, group_key, strict=True)),
                plot_suffix="__" + "_".join(f"{key}-{value}" for key, value in zip(group_by, group_key, strict=True)),
            )
            for group_key, group_records in groups.items()
        }

    return _analyze_hypers_group(
        records,
        target_hyperparameter=target_hyperparameter,
        experiment_slug=experiment_slug,
        verbose=verbose,
        group_label=None,
        plot_suffix="",
    )


@dataclass(frozen=True)
class _HyperparameterStatistics:
    winning_value: Any
    raw_mean: float
    corrected_mean: float
    maximization_bias: float
    sensitivity_slice: dict[Any, float]
    sensitivity_best: dict[Any, float]
    sorted_values: list[Any]
    sorted_slice: list[float]
    sorted_best: list[float]
    correction_ci: tuple[float, float]


def _compute_hyperparameter_statistics(records: list[dict[str, Any]], *, target_hyperparameter: str) -> _HyperparameterStatistics:
    runs_by_val: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        runs_by_val.setdefault(record["hypers"].get(target_hyperparameter), []).append(record)

    raw_means = {v: float(np.mean([r["value"] for r in records])) for v, records in runs_by_val.items()}
    winning_val = max(raw_means, key=lambda k: raw_means[k])
    raw_winner_mean = raw_means[winning_val]

    rng = np.random.default_rng(12345)
    bootstrap_winner_means = []
    unique_vals = list(runs_by_val.keys())
    for _ in range(1000):
        resampled_means = {}
        for val in unique_vals:
            records = runs_by_val[val]
            n = len(records)
            resampled_idx = rng.choice(n, size=n, replace=True)
            resampled_means[val] = np.mean([records[i]["value"] for i in resampled_idx])
        resampled_winner = max(resampled_means, key=lambda k: resampled_means[k])
        bootstrap_winner_means.append(raw_means[resampled_winner])

    corrected_mean = float(np.mean(bootstrap_winner_means))
    ci_low, ci_high = np.percentile(bootstrap_winner_means, [2.5, 97.5])
    maximization_bias = float(raw_winner_mean - corrected_mean)

    best_config = runs_by_val[winning_val][0]["hypers"]
    nuisance_keys = [k for k in best_config.keys() if k != target_hyperparameter and k != "seed"]
    slice_perf = {}
    best_perf = {}
    for val, records in runs_by_val.items():
        best_perf[val] = float(np.mean([r["value"] for r in records]))
        slice_records = [
            r for r in records
            if all(r["hypers"].get(k) == best_config.get(k) for k in nuisance_keys)
        ]
        slice_perf[val] = (
            float(np.mean([r["value"] for r in slice_records]))
            if slice_records
            else best_perf[val]
        )

    sorted_keys = sorted(unique_vals)
    return _HyperparameterStatistics(
        winning_value=winning_val,
        raw_mean=raw_winner_mean,
        corrected_mean=corrected_mean,
        maximization_bias=maximization_bias,
        sensitivity_slice=slice_perf,
        sensitivity_best=best_perf,
        sorted_values=sorted_keys,
        sorted_slice=[slice_perf[k] for k in sorted_keys],
        sorted_best=[best_perf[k] for k in sorted_keys],
        correction_ci=(float(ci_low), float(ci_high)),
    )


def _analyze_hypers_group(
    records: list[dict[str, Any]],
    *,
    target_hyperparameter: str,
    experiment_slug: str,
    verbose: bool,
    group_label: str | None,
    plot_suffix: str,
) -> HyperparameterSensitivityReport:
    """Run the maximization-bias-corrected sensitivity analysis over one set of run records."""
    statistics = _compute_hyperparameter_statistics(
        records,
        target_hyperparameter=target_hyperparameter,
    )
    winning_val = statistics.winning_value
    raw_winner_mean = statistics.raw_mean
    corrected_mean = statistics.corrected_mean
    maximization_bias = statistics.maximization_bias
    slice_perf = statistics.sensitivity_slice
    best_perf = statistics.sensitivity_best
    sorted_keys = statistics.sorted_values
    slice_perf_sorted = statistics.sorted_slice
    best_perf_sorted = statistics.sorted_best
    ci_low, ci_high = statistics.correction_ci

    analysis_dir = Path("results/analysis") / experiment_slug
    plot_path = analysis_dir / f"hyper_sensitivity{plot_suffix}.png"
    plot_sensitivity(target_hyperparameter, sorted_keys, slice_perf_sorted, best_perf_sorted, plot_path)

    report = HyperparameterSensitivityReport(
        target_hyperparameter=target_hyperparameter,
        winning_value=winning_val,
        raw_mean=raw_winner_mean,
        corrected_mean=corrected_mean,
        maximization_bias=maximization_bias,
        sensitivity_slice=slice_perf,
        sensitivity_best=best_perf,
        sensitivity_plot_path=plot_path,
    )

    if verbose:
        console = Console()
        console.print()
        subtitle = f"Experiment: {experiment_slug}" + (f" | Group: {group_label}" if group_label else "")
        console.print(
            Panel(
                Text(f"📊 HYPERPARAMETER ANALYSIS: {target_hyperparameter}", style="bold cyan"),
                subtitle=subtitle,
            )
        )

        table = Table(title="Sensitivity Matrix", show_header=True, header_style="bold magenta")
        table.add_column("Value", style="dim")
        table.add_column("Best Achievable Mean", justify="right")
        table.add_column("Optimal Slice Mean", justify="right")
        for val in sorted_keys:
            table.add_row(str(val), f"{best_perf[val]:.2f}", f"{slice_perf[val]:.2f}")
        console.print(table)

        bias_text = Panel(
            Text.assemble(
                ("Raw Winner Mean       : ", "dim"), (f"{raw_winner_mean:.2f} (value: {winning_val})\n"),
                ("Corrected Winner Mean : ", "bold green"), (f"{corrected_mean:.2f}\n"),
                ("Maximization Bias     : ", "bold red"), (f"+{maximization_bias:.2f}\n"),
                ("Bootstrap 95% CI      : ", "dim"), (f"[{ci_low:.2f}, {ci_high:.2f}]\n\n"),
                ("Pedagogical Interpretation:\n", "bold"),
                (
                    "Selecting the hyperparameter configuration with the highest raw average results "
                    "in Maximization Bias. This bias represents how much we have overestimated the algorithm's "
                    "true performance by selecting the 'lucky' configuration. The Corrected Mean represents "
                    "a realistic expectation of performance on a fresh set of random seeds."
                ),
            ),
            title="Maximization Bias Correction Log",
        )
        console.print(bias_text)

        console.print(
            Panel(
                Text(f"Saved sensitivity visual artifact to: {plot_path}", style="italic green"),
                title="Artifact Outputs Panel",
            )
        )

    return report


def _load_bakeoff_data(
    db_path: Path,
    experiment_slug: str,
    algorithms: list[str],
    environments: list[str],
    metric_name: str,
) -> tuple[dict[tuple[str, str], dict[int, float]], dict[str, list[float]]]:
    with DatabaseManager(db_path) as database:
        database.initialize()
        exp_row = database.get_experiment(experiment_slug)
        if exp_row is None:
            raise ValueError(f"Unknown experiment {experiment_slug!r}")
        runs = database.list_runs(exp_row.id)

    data_by_group: dict[tuple[str, str], dict[int, float]] = {}
    all_scores_by_env: dict[str, list[float]] = {env: [] for env in environments}
    with DatabaseManager(db_path) as database:
        for run in runs:
            latest_exec = database.get_latest_completed_execution_for_run(run.id)
            latest_art = database.get_latest_completed_artifacts_for_run(run.id)
            if latest_exec is None or latest_art is None:
                continue
            hyper_config = database.get_hyperparam_config(run.hyper_id)
            if hyper_config is None:
                continue
            hypers = json.loads(hyper_config.json_blob)
            algo = str(hypers.get("algorithm", ""))
            env = str(hypers.get("env_name", ""))
            if algo not in algorithms or env not in environments:
                continue
            metrics_db = _resolve_metrics_db_path(latest_art.root_path)
            metric_curve = _load_run_metric(metrics_db, run.id, metric_name)
            if metric_curve is None:
                continue
            data_by_group.setdefault((algo, env), {})[run.seed] = float(metric_curve[-1])
            all_scores_by_env[env].append(float(metric_curve[-1]))
    return data_by_group, all_scores_by_env


def compare_bakeoff(
    db_path: Path | str,
    experiment_slug: str,
    algorithms: list[str],
    environments: list[str],
    metric_name: str,
    verbose: bool = True,
) -> BenchmarkBakeoffReport:
    """Compare algorithms with ECDF normalization and a listwise-deleted Friedman test."""
    if len(algorithms) < 3:
        raise ValueError(
            f"compare_bakeoff requires at least 3 algorithms for the omnibus Friedman test, got "
            f"{len(algorithms)} ({algorithms!r}). For a two-condition comparison, use compare_pairwise instead."
        )
    db_path = Path(db_path)
    data_by_group, all_scores_by_env = _load_bakeoff_data(
        db_path,
        experiment_slug,
        algorithms,
        environments,
        metric_name,
    )

    # 1. Apply ECDF normalization per environment
    # ECDF score = fraction of all scores in pool that are less than raw score
    ecdf_normalized: dict[tuple[str, str], dict[int, float]] = {}
    for (algo, env), seed_map in data_by_group.items():
        pool = sorted(all_scores_by_env[env])
        if not pool:
            continue
        n_pool = len(pool)
        for seed, val in seed_map.items():
            rank_sum = sum(1 for score in pool if score < val)
            ecdf_normalized.setdefault((algo, env), {})[seed] = rank_sum / n_pool

    # Aggregate ECDF means per algorithm per env
    ecdf_means: dict[str, dict[str, float]] = {algo: {} for algo in algorithms}
    for algo in algorithms:
        for env in environments:
            seeds = ecdf_normalized.get((algo, env), {})
            if seeds:
                ecdf_means[algo][env] = float(np.mean(list(seeds.values())))
            else:
                ecdf_means[algo][env] = float("nan")

    # 2. Non-Parametric Omnibus Friedman Test
    # Rows are environment-seed indices and columns are algorithms.
    # Rows will represent environment-seed combinations.
    row_keys = []
    for env in environments:
        all_seeds_for_env = set()
        for algo in algorithms:
            all_seeds_for_env.update(ecdf_normalized.get((algo, env), {}).keys())
        for seed in sorted(all_seeds_for_env):
            row_keys.append((env, seed))

    matrix = []
    has_missing = False
    for env, seed in row_keys:
        row = []
        for algo in algorithms:
            val = ecdf_normalized.get((algo, env), {}).get(seed)
            if val is None:
                has_missing = True
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix, dtype=np.float64)

    valid_rows = [r for r in matrix if not np.isnan(r).any()]
    n_complete = len(valid_rows)
    if n_complete < 3:
        raise ValueError(
            f"Insufficient complete bakeoff data: need at least 3 rows, got {n_complete}"
        )

    friedman_matrix = np.array(valid_rows)
    res = stats.friedmanchisquare(*[friedman_matrix[:, i] for i in range(len(algorithms))])
    p_val = float(res.pvalue)
    stat = float(res.statistic)
    if has_missing:
        test_name = "Friedman Test (Listwise Deleted for Missing Data)"
        justification = (
            "Selected Friedman test with listwise deletion because missing values were present; "
            "rows containing missing data were excluded from the omnibus check."
        )
    else:
        test_name = "Friedman Test"
        justification = (
            "Selected Friedman test because data is complete (no missing environment-seed points)."
        )

    # 3. Post-hoc pairwise testing with Wilcoxon and Bonferroni corrections
    posthoc: dict[str, dict[str, float]] = {algo: {other: 1.0 for other in algorithms} for algo in algorithms}
    for i, algo_a in enumerate(algorithms):
        for j, algo_b in enumerate(algorithms):
            if i >= j:
                continue
            # Get paired observations
            paired_a = []
            paired_b = []
            for env, seed in row_keys:
                va = ecdf_normalized.get((algo_a, env), {}).get(seed)
                vb = ecdf_normalized.get((algo_b, env), {}).get(seed)
                if va is not None and vb is not None:
                    paired_a.append(va)
                    paired_b.append(vb)
            if len(paired_a) >= 3:
                res_w = stats.wilcoxon(paired_a, paired_b)
                raw_p = float(res_w.pvalue)
                # Bonferroni correction
                num_comparisons = (len(algorithms) * (len(algorithms) - 1)) / 2
                adjusted_p = min(1.0, raw_p * num_comparisons)
                posthoc[algo_a][algo_b] = adjusted_p
                posthoc[algo_b][algo_a] = adjusted_p

    # 4. Plot ECDF Curves
    # Construct curve values: sorted ECDF values and their cumulative proportions
    ecdf_curves = {}
    for algo in algorithms:
        all_vals = []
        for env in environments:
            all_vals.extend(ecdf_normalized.get((algo, env), {}).values())
        if all_vals:
            sorted_vals = sorted(all_vals)
            proportions = [i / len(sorted_vals) for i in range(1, len(sorted_vals) + 1)]
            ecdf_curves[algo] = (sorted_vals, proportions)

    analysis_dir = Path("results/analysis") / experiment_slug
    plot_path = analysis_dir / "ecdf_comparison.png"
    if ecdf_curves:
        plot_ecdf(ecdf_curves, plot_path)
    else:
        plot_path = None

    omnibus_details = StatisticalTestDetails(
        test_name=test_name,
        statistic=stat,
        p_value=p_val,
        effect_size=None,
        is_significant=p_val < 0.05,
        assumptions={"has_missing": has_missing},
        justification=justification,
        intermediate_tests={},
    )

    report = BenchmarkBakeoffReport(
        algorithms=algorithms,
        environments=environments,
        ecdf_scores=ecdf_means,
        omnibus_details=omnibus_details,
        posthoc_matrix=posthoc,
        ecdf_plot_path=plot_path,
    )

    if verbose:
        console = Console()
        console.print()
        console.print(
            Panel(
                Text(f"🏆 BENCHMARK BAKEOFF REPORT: {report.algorithms}", style="bold cyan"),
                subtitle=f"Experiment: {experiment_slug}",
            )
        )

        table = Table(title="ECDF-Normalized Performance Matrix", show_header=True, header_style="bold magenta")
        table.add_column("Algorithm")
        for env in environments:
            table.add_column(env, justify="right")
        for algo in algorithms:
            row = [algo]
            for env in environments:
                row.append(f"{ecdf_means[algo][env]:.2f}")
            table.add_row(*row)
        console.print(table)

        just_panel = Panel(
            Text.assemble(
                ("Omnibus Test Result:\n", "bold"),
                (f"• Selected Test: {omnibus_details.test_name}\n", "yellow"),
                (f"• Statistic: {omnibus_details.statistic:.4f}\n"),
                (f"• p-value: {omnibus_details.p_value:.4f}\n"),
                (f"• Justification: {omnibus_details.justification}\n\n", "italic"),
                ("Pedagogical Interpretation:\n", "bold"),
                (
                    "Raw scores cannot be averaged across different environments because they reside "
                    "on different numerical scales. ECDF normalization maps scores to their relative percentile "
                    "ranks across all algorithms tested on that specific task, making averages statistically valid."
                ),
            ),
            title="Omnibus Methodology Justification",
        )
        console.print(just_panel)

        if plot_path:
            console.print(
                Panel(
                    Text(f"Saved ECDF curves visual artifact to: {plot_path}", style="italic green"),
                    title="Artifact Outputs Panel",
                )
            )

    return report
