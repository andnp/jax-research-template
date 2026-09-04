"""Automated statistical reporting API for RL experiments.

Implements pairwise comparisons, hyperparameter sensitivity analysis, and benchmark bakeoffs
with pedagogical explanations and diagnostic plotting.
"""

from __future__ import annotations

import json
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

from research_analysis.bootstrap import bootstrap_difference_ci
from research_analysis.comparison import ComparisonSpec
from research_analysis.experiment import load_experiment_metrics
from research_analysis.permutation import run_permutation_test
from research_analysis.schema import AnalysisSchema


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
    estimand: str
    pairing: str
    alpha: float
    n_resamples: int
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
    schema: AnalysisSchema | None,
    spec: ComparisonSpec,
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
        schema=schema,
    )
    missing_keys = [key for key in spec.pairing_keys if key not in all_metrics.columns]
    if missing_keys:
        raise ValueError(f"Pairing keys are absent from loaded metrics: {missing_keys}")
    final_per_run = all_metrics.group_by(["condition_name", *spec.pairing_keys]).agg(pl.col("value").last())
    rows_a = final_per_run.filter(pl.col("condition_name") == condition_a).sort("seed")
    rows_b = final_per_run.filter(pl.col("condition_name") == condition_b).sort("seed")
    if rows_a.is_empty() or rows_b.is_empty():
        raise ValueError(
            f"No completed runs found for condition_name={condition_a!r} and condition_name={condition_b!r}"
        )

    keys = list(spec.pairing_keys)
    key_a = rows_a.select(keys)
    key_b = rows_b.select(keys)
    same_keys = set(key_a.rows()) == set(key_b.rows())
    is_paired = spec.pairing == "paired" or (spec.pairing == "auto" and same_keys)
    if not is_paired:
        return exp_row.name, rows_a["value"].to_numpy(), rows_b["value"].to_numpy(), False
    if not same_keys and spec.missingness == "error":
        raise ValueError("paired comparison requires identical pairing keys")
    paired = key_a.with_columns(rows_a["value"].alias("value_a")).join(
        key_b.with_columns(rows_b["value"].alias("value_b")), on=keys, how="inner"
    ).sort(keys)
    return exp_row.name, paired["value_a"].to_numpy(), paired["value_b"].to_numpy(), True


def _compute_pairwise_statistics(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    *,
    spec: ComparisonSpec,
    is_paired: bool,
) -> _PairwiseStatistics:
    pairing = "paired" if is_paired else "independent"
    permutation = run_permutation_test(
        vals_a,
        vals_b,
        estimand=spec.estimand,
        pairing=pairing,
        alpha=spec.alpha,
        n_resamples=spec.n_resamples,
        min_samples=spec.min_samples,
        rng=np.random.default_rng(spec.random_seed),
    )
    interval = bootstrap_difference_ci(
        vals_a,
        vals_b,
        estimand=spec.estimand,
        pairing=pairing,
        confidence=spec.confidence_level,
        n_resamples=spec.n_resamples,
        min_samples=spec.min_samples,
        rng=np.random.default_rng(spec.random_seed),
    )
    return _PairwiseStatistics(
        test_details=StatisticalTestDetails(
            test_name=f"{pairing.title()} permutation test",
            statistic=permutation.statistic,
            p_value=permutation.p_value,
            effect_size=permutation.statistic,
            is_significant=permutation.is_significant,
            assumptions={"paired": is_paired, "estimand": spec.estimand, "alpha": spec.alpha},
            justification="Permutation inference follows the declared comparison design without a normality pretest.",
            intermediate_tests={},
        ),
        difference_ci=(interval.ci_low, interval.ci_high),
    )


def _plot_pairwise(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    condition_a: str,
    condition_b: str,
    experiment_slug: str,
) -> Path:
    plot_path = Path("results/analysis") / experiment_slug / "ab_comparison.png"
    plot_distributions(vals_a, vals_b, condition_a, condition_b, plot_path)
    return plot_path


def _present_pairwise(
    report: ABComparisonReport,
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    condition_a: str,
    condition_b: str,
    metric_name: str,
) -> None:
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

    test_details = report.test_details
    console.print(
        Panel(
            Text.assemble(
                ("Declared Inference Design:\n", "bold"),
                (f"• Test: {test_details.test_name}\n", "yellow"),
                (f"• Estimand: {report.estimand}\n"),
                (f"• Pairing: {report.pairing}\n"),
                (f"• Alpha: {report.alpha:.3f}\n"),
                (f"• Resamples: {report.n_resamples}\n"),
                (f"• Justification: {test_details.justification}"),
            ),
            title="Methodology Justification Box",
        )
    )

    sig_style = "bold green" if test_details.is_significant else "bold red"
    sig_text = "Statistically Significant" if test_details.is_significant else "Not Statistically Significant"
    console.print(
        Panel(
            Text.assemble(
                ("Statistic       : ", "dim"), (f"{test_details.statistic:.4f}\n"),
                ("p-value         : ", "dim"), (f"{test_details.p_value:.4f} ", sig_style), (f"({sig_text})\n", sig_style),
                ("Difference Mean : ", "dim"), (f"{report.difference_in_means:.4f}\n"),
                ("BCa CI          : ", "dim"), (f"[{report.difference_ci[0]:.2f}, {report.difference_ci[1]:.2f}]\n\n"),
                ("Pedagogical Interpretation:\n", "bold"),
                (
                    f"There is a {report.test_details.p_value * 100:.2f}% chance that we would observe "
                    "a difference this large purely due to random variations if the two algorithms "
                    "were completely identical in performance."
                ),
            ),
            title="Findings & Interpretation",
        )
    )
    console.print(
        Panel(
            Text(f"Saved distribution visual artifact to: {report.distribution_plot_path}", style="italic green"),
            title="Artifact Outputs Panel",
        )
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
    schema: AnalysisSchema | None = None,
    spec: ComparisonSpec | None = None,
) -> ABComparisonReport:
    """Compare exactly two conditions using a declared statistical design."""
    if spec is None:
        if not 0 < confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        spec = ComparisonSpec(alpha=1.0 - confidence_level)
    elif confidence_level != 0.95:
        raise ValueError("pass confidence through spec when spec is provided")
    db_path = Path(db_path)
    experiment_name, vals_a, vals_b, is_paired = _load_pairwise_data(
        db_path,
        experiment_slug,
        condition_a,
        condition_b,
        metric_name,
        schema,
        spec,
    )

    pairwise_statistics = _compute_pairwise_statistics(
        vals_a,
        vals_b,
        spec=spec,
        is_paired=is_paired,
    )
    test_details = pairwise_statistics.test_details
    ci_low, ci_high = pairwise_statistics.difference_ci

    plot_path = _plot_pairwise(
        vals_a,
        vals_b,
        condition_a,
        condition_b,
        experiment_slug,
    )

    report = ABComparisonReport(
        experiment_name=experiment_name,
        condition_a=condition_a,
        condition_b=condition_b,
        metric_name=metric_name,
        estimand=spec.estimand,
        pairing="paired" if is_paired else "independent",
        alpha=spec.alpha,
        n_resamples=spec.n_resamples,
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        difference_in_means=float(np.mean(vals_a) - np.mean(vals_b)),
        difference_ci=(float(ci_low), float(ci_high)),
        test_details=test_details,
        distribution_plot_path=plot_path,
    )

    if verbose:
        _present_pairwise(
            report,
            vals_a,
            vals_b,
            condition_a,
            condition_b,
            metric_name,
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


def _plot_hypers(
    target_hyperparameter: str,
    sorted_keys: list[Any],
    slice_perf_sorted: list[float],
    best_perf_sorted: list[float],
    experiment_slug: str,
    plot_suffix: str,
) -> Path:
    plot_path = Path("results/analysis") / experiment_slug / f"hyper_sensitivity{plot_suffix}.png"
    plot_sensitivity(target_hyperparameter, sorted_keys, slice_perf_sorted, best_perf_sorted, plot_path)
    return plot_path


def _present_hypers(
    report: HyperparameterSensitivityReport,
    sorted_keys: list[Any],
    experiment_slug: str,
    group_label: str | None,
    correction_ci: tuple[float, float],
) -> None:
    console = Console()
    console.print()
    subtitle = f"Experiment: {experiment_slug}" + (f" | Group: {group_label}" if group_label else "")
    console.print(
        Panel(
            Text(f"📊 HYPERPARAMETER ANALYSIS: {report.target_hyperparameter}", style="bold cyan"),
            subtitle=subtitle,
        )
    )

    table = Table(title="Sensitivity Matrix", show_header=True, header_style="bold magenta")
    table.add_column("Value", style="dim")
    table.add_column("Best Achievable Mean", justify="right")
    table.add_column("Optimal Slice Mean", justify="right")
    for val in sorted_keys:
        table.add_row(str(val), f"{report.sensitivity_best[val]:.2f}", f"{report.sensitivity_slice[val]:.2f}")
    console.print(table)

    ci_low, ci_high = correction_ci
    console.print(
        Panel(
            Text.assemble(
                ("Raw Winner Mean       : ", "dim"), (f"{report.raw_mean:.2f} (value: {report.winning_value})\n"),
                ("Corrected Winner Mean : ", "bold green"), (f"{report.corrected_mean:.2f}\n"),
                ("Maximization Bias     : ", "bold red"), (f"+{report.maximization_bias:.2f}\n"),
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
    )
    console.print(
        Panel(
            Text(f"Saved sensitivity visual artifact to: {report.sensitivity_plot_path}", style="italic green"),
            title="Artifact Outputs Panel",
        )
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

    plot_path = _plot_hypers(
        target_hyperparameter,
        sorted_keys,
        slice_perf_sorted,
        best_perf_sorted,
        experiment_slug,
        plot_suffix,
    )

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
        _present_hypers(
            report,
            sorted_keys,
            experiment_slug,
            group_label,
            statistics.correction_ci,
        )

    return report


def _load_bakeoff_data(
    db_path: Path,
    experiment_slug: str,
    algorithms: list[str],
    environments: list[str],
    metric_name: str,
    schema: AnalysisSchema | None,
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
            algo = (
                schema.condition_name(hypers)
                if schema is not None
                else str(hypers.get("algorithm", ""))
            )
            env = (
                schema.environment_name(hypers)
                if schema is not None
                else str(hypers.get("env_name", ""))
            )
            if algo not in algorithms or env not in environments:
                continue
            metrics_db = _resolve_metrics_db_path(latest_art.root_path)
            metric_curve = _load_run_metric(metrics_db, run.id, metric_name)
            if metric_curve is None:
                continue
            data_by_group.setdefault((algo, env), {})[run.seed] = float(metric_curve[-1])
            all_scores_by_env[env].append(float(metric_curve[-1]))
    return data_by_group, all_scores_by_env


@dataclass(frozen=True)
class _BakeoffStatistics:
    ecdf_scores: dict[str, dict[str, float]]
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float]]
    ecdf_curves: dict[str, tuple[list[float], list[float]]]


def _compute_bakeoff_statistics(
    data_by_group: dict[tuple[str, str], dict[int, float]],
    all_scores_by_env: dict[str, list[float]],
    algorithms: list[str],
    environments: list[str],
) -> _BakeoffStatistics:
    ecdf_normalized: dict[tuple[str, str], dict[int, float]] = {}
    for (algo, env), seed_map in data_by_group.items():
        pool = sorted(all_scores_by_env[env])
        if not pool:
            continue
        n_pool = len(pool)
        for seed, val in seed_map.items():
            rank_sum = sum(1 for score in pool if score < val)
            ecdf_normalized.setdefault((algo, env), {})[seed] = rank_sum / n_pool

    ecdf_means: dict[str, dict[str, float]] = {algo: {} for algo in algorithms}
    for algo in algorithms:
        for env in environments:
            seeds = ecdf_normalized.get((algo, env), {})
            ecdf_means[algo][env] = float(np.mean(list(seeds.values()))) if seeds else float("nan")

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
        raise ValueError(f"Insufficient complete bakeoff data: need at least 3 rows, got {n_complete}")

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
        justification = "Selected Friedman test because data is complete (no missing environment-seed points)."

    posthoc: dict[str, dict[str, float]] = {algo: {other: 1.0 for other in algorithms} for algo in algorithms}
    for i, algo_a in enumerate(algorithms):
        for j, algo_b in enumerate(algorithms):
            if i >= j:
                continue
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
                num_comparisons = (len(algorithms) * (len(algorithms) - 1)) / 2
                adjusted_p = min(1.0, raw_p * num_comparisons)
                posthoc[algo_a][algo_b] = adjusted_p
                posthoc[algo_b][algo_a] = adjusted_p

    ecdf_curves = {}
    for algo in algorithms:
        all_vals = []
        for env in environments:
            all_vals.extend(ecdf_normalized.get((algo, env), {}).values())
        if all_vals:
            sorted_vals = sorted(all_vals)
            proportions = [i / len(sorted_vals) for i in range(1, len(sorted_vals) + 1)]
            ecdf_curves[algo] = (sorted_vals, proportions)

    return _BakeoffStatistics(
        ecdf_scores=ecdf_means,
        omnibus_details=StatisticalTestDetails(
            test_name=test_name,
            statistic=stat,
            p_value=p_val,
            effect_size=None,
            is_significant=p_val < 0.05,
            assumptions={"has_missing": has_missing},
            justification=justification,
            intermediate_tests={},
        ),
        posthoc_matrix=posthoc,
        ecdf_curves=ecdf_curves,
    )


def _plot_bakeoff(
    ecdf_curves: dict[str, tuple[list[float], list[float]]],
    experiment_slug: str,
) -> Path | None:
    plot_path = Path("results/analysis") / experiment_slug / "ecdf_comparison.png"
    if ecdf_curves:
        plot_ecdf(ecdf_curves, plot_path)
        return plot_path
    return None


def _present_bakeoff(
    report: BenchmarkBakeoffReport,
    environments: list[str],
    experiment_slug: str,
) -> None:
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
    for algo in report.algorithms:
        row = [algo]
        for env in environments:
            row.append(f"{report.ecdf_scores[algo][env]:.2f}")
        table.add_row(*row)
    console.print(table)

    omnibus_details = report.omnibus_details
    console.print(
        Panel(
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
    )

    if report.ecdf_plot_path:
        console.print(
            Panel(
                Text(f"Saved ECDF curves visual artifact to: {report.ecdf_plot_path}", style="italic green"),
                title="Artifact Outputs Panel",
            )
        )


def compare_bakeoff(
    db_path: Path | str,
    experiment_slug: str,
    algorithms: list[str],
    environments: list[str],
    metric_name: str,
    verbose: bool = True,
    schema: AnalysisSchema | None = None,
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
        schema,
    )

    statistics = _compute_bakeoff_statistics(
        data_by_group,
        all_scores_by_env,
        algorithms,
        environments,
    )
    ecdf_means = statistics.ecdf_scores
    omnibus_details = statistics.omnibus_details
    posthoc = statistics.posthoc_matrix
    ecdf_curves = statistics.ecdf_curves

    plot_path = _plot_bakeoff(ecdf_curves, experiment_slug)

    report = BenchmarkBakeoffReport(
        algorithms=algorithms,
        environments=environments,
        ecdf_scores=ecdf_means,
        omnibus_details=omnibus_details,
        posthoc_matrix=posthoc,
        ecdf_plot_path=plot_path,
    )

    if verbose:
        _present_bakeoff(report, environments, experiment_slug)

    return report
