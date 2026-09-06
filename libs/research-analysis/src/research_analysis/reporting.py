"""Automated statistical reporting API for RL experiments.

Implements pairwise comparisons, hyperparameter sensitivity analysis, and benchmark bakeoffs
with pedagogical explanations and diagnostic plotting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import polars as pl
from research_plot.plots import plot_distributions, plot_ecdf, plot_sensitivity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from research_analysis.aggregates import (
    ReferenceRange,
    interquartile_mean,
    normalize_scores,
    optimality_gap,
    performance_profile,
    probability_of_improvement,
)
from research_analysis.bootstrap import bootstrap_difference_ci
from research_analysis.comparison import ComparisonSpec
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.diagnostics.bakeoff import BakeoffWarning, BakeoffWarningCode
from research_analysis.multiplicity import holm_step_down
from research_analysis.permutation import run_permutation_test
from research_analysis.repeated_blocks import block_stratified_permutation_test
from research_analysis.tuning import _build_hyper_matrix, hierarchical_bootstrap_ci, hierarchical_bootstrap_difference_ci


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
    min_samples: int
    missingness: str
    mean_a: float
    mean_b: float
    difference_in_means: float
    estimate_a: float
    estimate_b: float
    difference_in_estimand: float
    difference_ci: tuple[float, float]
    test_details: StatisticalTestDetails
    distribution_plot_path: Path | None


@dataclass(frozen=True)
class HyperparameterConfigReport:
    """Bias-corrected estimate of the best hyperparameter configuration's true performance."""

    winning_config: dict[str, Any]
    raw_mean: float
    corrected_mean: float
    maximization_bias: float
    estimate_ci: tuple[float, float]
    confidence_level: float = 0.95


@dataclass(frozen=True)
class HyperparameterDifferenceReport:
    """Bias-corrected paired difference between two groups' best hyperparameter configurations."""

    minuend_config: dict[str, Any]
    subtrahend_config: dict[str, Any]
    mean_diff: float
    estimate_ci: tuple[float, float]
    confidence_level: float = 0.95


@dataclass(frozen=True)
class HyperparameterSliceReport:
    """Sensitivity curve for one hyperparameter: best-achievable performance vs. a champion-pinned slice."""

    vary: str
    values: list[Any]
    best_performance: list[float]
    slice_performance: list[float]
    plot_path: Path | None


@dataclass(frozen=True)
class BenchmarkBakeoffReport:
    algorithms: list[str]
    environments: list[str]
    normalized_scores: dict[str, dict[str, float]]
    performance_profiles: dict[str, dict[float, float]]
    probability_of_improvement: dict[str, dict[str, float | None]]
    optimality_gaps: dict[str, float]
    dispersion: dict[str, float]
    lower_tail: dict[str, float]
    supplementary_iqm: dict[str, float]
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float | None]]
    aggregate_plot_path: Path | None


@dataclass(frozen=True)
class _PairwiseStatistics:
    test_details: StatisticalTestDetails
    difference_ci: tuple[float, float]


# ── Entry Points ─────────────────────────────────────────────────────────────


def _load_pairwise_data(
    dataset: ExperimentDataset,
    condition_a: str,
    condition_b: str,
    metric_name: str,
    spec: ComparisonSpec,
) -> tuple[str, np.ndarray, np.ndarray, bool]:
    all_metrics = dataset.metrics.filter(pl.col("metric") == metric_name).sort("step")
    missing_keys = [key for key in spec.pairing_keys if key not in all_metrics.columns]
    if missing_keys:
        raise ValueError(f"Pairing keys are absent from loaded metrics: {missing_keys}")
    if spec.missingness == "blockwise":
        raise ValueError("pairwise comparisons do not support missingness='blockwise'")
    keys = list(spec.pairing_keys)
    final_per_run = all_metrics.group_by(["condition_name", *spec.pairing_keys]).agg(pl.col("value").last())
    rows_a = final_per_run.filter(pl.col("condition_name") == condition_a).sort(keys)
    rows_b = final_per_run.filter(pl.col("condition_name") == condition_b).sort(keys)
    if rows_a.is_empty() or rows_b.is_empty():
        raise ValueError(
            f"No completed runs found for condition_name={condition_a!r} and condition_name={condition_b!r}"
        )

    key_a = rows_a.select(keys)
    key_b = rows_b.select(keys)
    keys_a = set(key_a.rows())
    keys_b = set(key_b.rows())
    same_keys = keys_a == keys_b
    partial_overlap = bool(keys_a & keys_b) and not same_keys
    if spec.pairing == "auto" and partial_overlap and spec.missingness == "error":
        raise ValueError(
            "automatic pairing found partially overlapping pairing keys; "
            "use missingness='complete_case' to pair the overlap explicitly"
        )
    is_paired = spec.pairing == "paired" or (spec.pairing == "auto" and (same_keys or partial_overlap))
    if not is_paired:
        return dataset.experiment_name, rows_a["value"].to_numpy(), rows_b["value"].to_numpy(), False
    if not same_keys and spec.missingness != "complete_case":
        raise ValueError(
            "paired comparison requires identical pairing keys unless missingness='complete_case'"
        )
    paired = key_a.with_columns(rows_a["value"].alias("value_a")).join(
        key_b.with_columns(rows_b["value"].alias("value_b")), on=keys, how="inner"
    ).sort(keys)
    return dataset.experiment_name, paired["value_a"].to_numpy(), paired["value_b"].to_numpy(), True


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
            assumptions={
                "paired": is_paired,
                "estimand": spec.estimand,
                "alpha": spec.alpha,
                "n_resamples": spec.n_resamples,
                "min_samples": spec.min_samples,
                "missingness": spec.missingness,
            },
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
    table.add_column(f"{report.estimand.title()} estimate", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_row(
        condition_a,
        str(len(vals_a)),
        f"{report.mean_a:.2f}",
        f"{report.estimate_a:.2f}",
        f"{np.std(vals_a, ddof=1):.2f}",
    )
    table.add_row(
        condition_b,
        str(len(vals_b)),
        f"{report.mean_b:.2f}",
        f"{report.estimate_b:.2f}",
        f"{np.std(vals_b, ddof=1):.2f}",
    )
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
                (f"Difference {report.estimand.title()} : ", "dim"),
                (f"{report.difference_in_estimand:.4f}\n"),
                (f"BCa {1.0 - report.alpha:.0%} CI       : ", "dim"),
                (f"[{report.difference_ci[0]:.2f}, {report.difference_ci[1]:.2f}]\n\n"),
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


def _load_dataset_hyperparameter_records(
    dataset: ExperimentDataset,
    hyperparameter_keys: Sequence[str],
    metric_name: str,
) -> list[dict[str, Any]]:
    """Return final metric values joined to the dataset's declared sweep axes and condition.

    A requested key equal to ``dataset.dimensions.condition`` is read from the
    physical ``condition_name`` column, since the condition dimension is never
    itself a declared ``SweepAxis``.
    """
    run_columns = {
        key: ("condition_name" if key == dataset.dimensions.condition else key) for key in hyperparameter_keys
    }
    missing_keys = [key for key, column in run_columns.items() if column not in dataset.runs.columns]
    if missing_keys:
        raise ValueError(f"No declared sweep axes found for {missing_keys!r}")

    final_metrics = (
        dataset.metrics.filter(pl.col("metric") == metric_name)
        .sort(["run_id", "execution_id", "step"])
        .group_by(["run_id", "execution_id"])
        .agg(pl.col("value").last().alias("value"))
    )
    joined = final_metrics.join(
        dataset.runs.select(["run_id", "seed", *run_columns.values()]), on=["run_id"], how="inner"
    )
    records: list[dict[str, Any]] = []
    for row in joined.iter_rows(named=True):
        hypers = {key: row[column] for key, column in run_columns.items()}
        records.append({"run_id": row["run_id"], "seed": row["seed"], "value": row["value"], "hypers": hypers})

    if not records:
        raise ValueError(f"No completed runs found containing hyperparameters {list(hyperparameter_keys)!r}")
    return records


def compare_pairwise(
    dataset: ExperimentDataset,
    condition_a: str,
    condition_b: str,
    metric_name: str,
    confidence_level: float = 0.95,
    verbose: bool = True,
    spec: ComparisonSpec | None = None,
) -> ABComparisonReport:
    """Compare exactly two conditions using a declared statistical design."""
    if spec is None:
        if not 0 < confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        spec = ComparisonSpec(alpha=1.0 - confidence_level)
    elif confidence_level != 0.95:
        raise ValueError("pass confidence through spec when spec is provided")
    experiment_name, vals_a, vals_b, is_paired = _load_pairwise_data(
        dataset,
        condition_a,
        condition_b,
        metric_name,
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
        dataset.experiment_name,
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
        min_samples=spec.min_samples,
        missingness=spec.missingness,
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        difference_in_means=float(np.mean(vals_a) - np.mean(vals_b)),
        estimate_a=float(np.mean(vals_a) if spec.estimand == "mean" else np.median(vals_a)),
        estimate_b=float(np.mean(vals_b) if spec.estimand == "mean" else np.median(vals_b)),
        difference_in_estimand=pairwise_statistics.test_details.statistic,
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
def best_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    group_by: None = None,
    *,
    difference_between: None = None,
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> HyperparameterConfigReport: ...


@overload
def best_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    group_by: list[str],
    *,
    difference_between: None = None,
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> dict[tuple[Any, ...], HyperparameterConfigReport]: ...


@overload
def best_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    group_by: list[str],
    *,
    difference_between: tuple[Any, Any],
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> HyperparameterDifferenceReport: ...


def best_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    group_by: list[str] | None = None,
    *,
    difference_between: tuple[Any, Any] | None = None,
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> HyperparameterConfigReport | dict[tuple[Any, ...], HyperparameterConfigReport] | HyperparameterDifferenceReport:
    """Find the bias-corrected best hyperparameter configuration.

    Jointly optimizes every declared sweep axis not named in ``group_by``. When
    ``group_by`` names one or more hyperparameter keys, the search is performed
    separately for each distinct combination of those keys' values, and a mapping
    from the group key tuple to its own ``HyperparameterConfigReport`` is returned
    instead of a single pooled report. An empty or absent ``group_by`` jointly
    optimizes every declared sweep axis and returns one report for the whole grid.
    Set ``higher_is_better=False`` for error-like metrics where the champion is the
    minimizer.

    When ``difference_between`` is given (as ``(minuend_value, subtrahend_value)``),
    ``group_by`` must name exactly one axis, and the paired difference between the two
    named values' own champion configurations is returned instead, correlated by
    shared seed so that common sampling noise cancels out.

    ``group_by``/``difference_between`` may also name the dataset's condition
    dimension (e.g. an algorithm arm), in addition to declared sweep axes.
    """
    declared_axes = {axis.name for axis in dataset.dimensions.sweep_axes}
    groupable_axes = declared_axes | {dataset.dimensions.condition}
    group_keys = list(group_by or ())
    undeclared_axes = set(group_keys) - groupable_axes
    if undeclared_axes:
        raise ValueError(
            "hyperparameter analysis requires declared typed sweep axes or the condition dimension; "
            f"undeclared keys: {sorted(undeclared_axes)}"
        )

    if difference_between is not None:
        if len(group_keys) != 1:
            raise ValueError("difference_between requires group_by to name exactly one axis")
        return _best_hypers_difference(
            dataset,
            metric_name,
            group_axis=group_keys[0],
            minuend=difference_between[0],
            subtrahend=difference_between[1],
            declared_axes=declared_axes,
            higher_is_better=higher_is_better,
            verbose=verbose,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            min_samples=min_samples,
            random_seed=random_seed,
        )

    free_axes = sorted(declared_axes - set(group_keys))
    if not free_axes:
        raise ValueError("need at least one swept hyperparameter outside group_by to optimize")

    records = _load_dataset_hyperparameter_records(dataset, sorted(declared_axes | set(group_keys)), metric_name)

    if group_keys:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in records:
            group_key = tuple(record["hypers"].get(key) for key in group_keys)
            groups.setdefault(group_key, []).append(record)

        return {
            group_key: _best_hypers_group(
                group_records,
                free_axes=free_axes,
                experiment_slug=dataset.experiment_name,
                verbose=verbose,
                group_label=", ".join(f"{key}={value}" for key, value in zip(group_keys, group_key, strict=True)),
                higher_is_better=higher_is_better,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                min_samples=min_samples,
                random_seed=random_seed,
            )
            for group_key, group_records in groups.items()
        }

    return _best_hypers_group(
        records,
        free_axes=free_axes,
        experiment_slug=dataset.experiment_name,
        verbose=verbose,
        group_label=None,
        higher_is_better=higher_is_better,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        min_samples=min_samples,
        random_seed=random_seed,
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
                    "true performance by selecting the 'lucky' configuration. The Corrected Mean represents "
                    "a realistic expectation of performance on a fresh set of random seeds."
                ),
            ),
            title="Maximization Bias Correction Log",
        )
    )


def _raw_best_config(records: list[dict[str, Any]], free_axes: list[str]) -> tuple[dict[str, Any], float]:
    """Return the raw-mean-maximizing combination of ``free_axes`` and its own raw mean."""
    runs_by_config: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record["hypers"][axis] for axis in free_axes)
        runs_by_config.setdefault(key, []).append(record)

    raw_means = {key: float(np.mean([r["value"] for r in recs])) for key, recs in runs_by_config.items()}

    def raw_mean_of(k: tuple[Any, ...]) -> float:
        return raw_means[k]

    winning_key = max(raw_means, key=raw_mean_of)
    return dict(zip(free_axes, winning_key, strict=True)), raw_means[winning_key]


def _negate_record_values(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flip the sign of every record's metric value, so a maximizer finds the minimizer."""
    return [{**record, "value": -record["value"]} for record in records]


def _best_hypers_group(
    records: list[dict[str, Any]],
    *,
    free_axes: list[str],
    experiment_slug: str,
    verbose: bool,
    group_label: str | None,
    higher_is_better: bool,
    confidence_level: float,
    n_resamples: int,
    min_samples: int,
    random_seed: int,
) -> HyperparameterConfigReport:
    """Run the maximization-bias-corrected joint hyperparameter search over one set of run records."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")

    search_records = records if higher_is_better else _negate_record_values(records)
    winning_config, raw_winner_mean = _raw_best_config(search_records, free_axes)

    matrix, _matrix_keys, matrix_seeds = _build_hyper_matrix(search_records, free_axes)
    if len(matrix_seeds) < min_samples:
        raise ValueError(f"need at least {min_samples} seeds shared by every hyperparameter configuration")
    n_runs = matrix.shape[1]

    bootstrap = hierarchical_bootstrap_ci(
        matrix,
        n_runs,
        n_runs,
        confidence=confidence_level,
        n_resamples=n_resamples,
        seed=random_seed,
    )
    corrected_mean = bootstrap.mean
    interval = (bootstrap.ci_low, bootstrap.ci_high)
    maximization_bias = float(raw_winner_mean - corrected_mean)

    if not higher_is_better:
        raw_winner_mean = -raw_winner_mean
        corrected_mean = -corrected_mean
        interval = (-interval[1], -interval[0])

    report = HyperparameterConfigReport(
        winning_config=winning_config,
        raw_mean=raw_winner_mean,
        corrected_mean=corrected_mean,
        maximization_bias=maximization_bias,
        estimate_ci=interval,
        confidence_level=confidence_level,
    )

    if verbose:
        _present_best_hypers(report, experiment_slug, group_label)

    return report


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


def _best_hypers_difference(
    dataset: ExperimentDataset,
    metric_name: str,
    *,
    group_axis: str,
    minuend: Any,
    subtrahend: Any,
    declared_axes: set[str],
    higher_is_better: bool,
    verbose: bool,
    confidence_level: float,
    n_resamples: int,
    min_samples: int,
    random_seed: int,
) -> HyperparameterDifferenceReport:
    """Paired difference between two group_axis values' own bias-corrected champions."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")

    free_axes = sorted(declared_axes - {group_axis})
    if not free_axes:
        raise ValueError("need at least one swept hyperparameter outside group_by to optimize")

    records = _load_dataset_hyperparameter_records(dataset, sorted(declared_axes | {group_axis}), metric_name)
    records_a = [record for record in records if record["hypers"].get(group_axis) == minuend]
    records_b = [record for record in records if record["hypers"].get(group_axis) == subtrahend]
    if not records_a:
        raise ValueError(f"no runs found with {group_axis}={minuend!r}")
    if not records_b:
        raise ValueError(f"no runs found with {group_axis}={subtrahend!r}")

    if not higher_is_better:
        records_a = _negate_record_values(records_a)
        records_b = _negate_record_values(records_b)

    minuend_config, _minuend_raw_mean = _raw_best_config(records_a, free_axes)
    subtrahend_config, _subtrahend_raw_mean = _raw_best_config(records_b, free_axes)

    matrix_a, _keys_a, seeds_a = _build_hyper_matrix(records_a, free_axes)
    matrix_b, _keys_b, seeds_b = _build_hyper_matrix(records_b, free_axes)
    common_seeds = sorted(set(seeds_a) & set(seeds_b))
    if len(common_seeds) < min_samples:
        raise ValueError(f"need at least {min_samples} seeds shared by both compared groups")
    matrix_a = matrix_a[:, [seeds_a.index(seed) for seed in common_seeds]]
    matrix_b = matrix_b[:, [seeds_b.index(seed) for seed in common_seeds]]
    n_runs = matrix_a.shape[1]

    diff = hierarchical_bootstrap_difference_ci(
        matrix_a,
        matrix_b,
        n_runs,
        n_runs,
        confidence=confidence_level,
        n_resamples=n_resamples,
        seed=random_seed,
    )
    mean_diff = diff.mean_diff
    interval = (diff.ci_low, diff.ci_high)
    if not higher_is_better:
        mean_diff = -mean_diff
        interval = (-interval[1], -interval[0])

    report = HyperparameterDifferenceReport(
        minuend_config=minuend_config,
        subtrahend_config=subtrahend_config,
        mean_diff=mean_diff,
        estimate_ci=interval,
        confidence_level=confidence_level,
    )

    if verbose:
        _present_best_hypers_difference(report, dataset.experiment_name)

    return report


@overload
def slice_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    vary: str,
    group_by: None = None,
    *,
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> HyperparameterSliceReport: ...


@overload
def slice_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    vary: str,
    group_by: list[str],
    *,
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> dict[tuple[Any, ...], HyperparameterSliceReport]: ...


def slice_hypers(
    dataset: ExperimentDataset,
    metric_name: str,
    vary: str,
    group_by: list[str] | None = None,
    *,
    higher_is_better: bool = True,
    verbose: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    random_seed: int = 0,
) -> HyperparameterSliceReport | dict[tuple[Any, ...], HyperparameterSliceReport]:
    """Sensitivity curve for one hyperparameter: best-achievable vs. champion-pinned slice.

    Composes ``best_hypers`` twice -- once without ``vary`` to find the configuration
    every other axis is pinned at for the slice curve, and once with ``vary`` added to
    ``group_by`` to get each of its values' own best-achievable performance -- rather than
    computing any new statistics. The pinned slice itself is a plain per-value mean, since
    nothing is being selected there. Set ``higher_is_better=False`` for error-like metrics.
    """
    declared_axes = {axis.name for axis in dataset.dimensions.sweep_axes}
    group_keys = list(group_by or ())
    if vary not in declared_axes:
        raise ValueError(f"{vary!r} is not a declared swept hyperparameter")
    if vary in group_keys:
        raise ValueError(f"{vary!r} cannot be both varied and grouped by")

    champions = best_hypers(
        dataset,
        metric_name,
        group_by=group_keys,
        higher_is_better=higher_is_better,
        verbose=False,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        min_samples=min_samples,
        random_seed=random_seed,
    )
    profiles = best_hypers(
        dataset,
        metric_name,
        group_by=[*group_keys, vary],
        higher_is_better=higher_is_better,
        verbose=False,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        min_samples=min_samples,
        random_seed=random_seed,
    )
    records = _load_dataset_hyperparameter_records(dataset, sorted(declared_axes | set(group_keys)), metric_name)

    if group_keys:
        assert isinstance(champions, dict)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in records:
            group_key = tuple(record["hypers"].get(key) for key in group_keys)
            groups.setdefault(group_key, []).append(record)

        return {
            group_key: _slice_hypers_group(
                group_records,
                vary=vary,
                champion=champions[group_key],
                profile_by_value={key[-1]: report for key, report in profiles.items() if key[:-1] == group_key},
                experiment_slug=dataset.experiment_name,
                group_label=", ".join(f"{key}={value}" for key, value in zip(group_keys, group_key, strict=True)),
                plot_suffix="__" + "_".join(f"{key}-{value}" for key, value in zip(group_keys, group_key, strict=True)),
                verbose=verbose,
            )
            for group_key, group_records in groups.items()
        }

    assert isinstance(champions, HyperparameterConfigReport)
    return _slice_hypers_group(
        records,
        vary=vary,
        champion=champions,
        profile_by_value={key[0]: report for key, report in profiles.items()},
        experiment_slug=dataset.experiment_name,
        group_label=None,
        plot_suffix="",
        verbose=verbose,
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
        table.add_row(str(value), f"{best:.2f}", f"{slice_val:.2f}")
    console.print(table)

    console.print(
        Panel(
            Text(f"Saved sensitivity visual artifact to: {report.plot_path}", style="italic green"),
            title="Artifact Outputs Panel",
        )
    )


def _slice_hypers_group(
    records: list[dict[str, Any]],
    *,
    vary: str,
    champion: HyperparameterConfigReport,
    profile_by_value: dict[Any, HyperparameterConfigReport],
    experiment_slug: str,
    group_label: str | None,
    plot_suffix: str,
    verbose: bool,
) -> HyperparameterSliceReport:
    pinned = {key: value for key, value in champion.winning_config.items() if key != vary}
    matching = [record for record in records if all(record["hypers"].get(k) == v for k, v in pinned.items())]

    slice_by_value: dict[Any, list[float]] = {}
    for record in matching:
        slice_by_value.setdefault(record["hypers"][vary], []).append(record["value"])

    sorted_values = sorted(profile_by_value)
    best_performance = [profile_by_value[value].raw_mean for value in sorted_values]
    slice_performance = [
        float(np.mean(slice_by_value[value])) if value in slice_by_value else profile_by_value[value].raw_mean
        for value in sorted_values
    ]

    plot_path = _plot_hypers(vary, sorted_values, slice_performance, best_performance, experiment_slug, plot_suffix)

    report = HyperparameterSliceReport(
        vary=vary,
        values=sorted_values,
        best_performance=best_performance,
        slice_performance=slice_performance,
        plot_path=plot_path,
    )

    if verbose:
        _present_slice_hypers(report, experiment_slug, group_label)

    return report


def boundary_warning(
    report: HyperparameterSliceReport,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    higher_is_better: bool = True,
) -> BakeoffWarning | None:
    """Flag a champion hyperparameter sitting at the edge of its swept grid while
    performance is still trending toward that edge over the grid's outer half, not
    merely plateaued there. A champion at the edge with an interior peak elsewhere
    (or a plateau right at the edge) is not flagged: only a still-climbing trend
    means the true optimum may lie outside the swept range.
    """
    values = report.values
    performance = report.best_performance
    if len(values) < 2:
        return None

    champion_idx = int(np.argmax(performance)) if higher_is_better else int(np.argmin(performance))
    if champion_idx == 0:
        edge_label, edge_value = "lower", values[0]
    elif champion_idx == len(values) - 1:
        edge_label, edge_value = "upper", values[-1]
    else:
        return None

    half = max(1, len(values) // 2)
    outer = performance[-half - 1 :] if edge_label == "upper" else performance[: half + 1]
    trend = np.diff(outer)
    if not higher_is_better:
        trend = -trend
    if edge_label == "lower":
        trend = -trend
    if float(np.mean(trend)) <= 0:
        return None

    return BakeoffWarning(
        code=BakeoffWarningCode.HYPERPARAMETER_AT_BOUNDARY,
        message=(
            f"{report.vary}={edge_value!r} sits at the {edge_label} edge of the swept grid "
            "and performance is still trending toward it; the true optimum may lie outside "
            "the swept range."
        ),
        condition=condition,
        cell=cell,
    )


def _load_bakeoff_data(
    dataset: ExperimentDataset,
    algorithms: list[str],
    environments: list[str],
    metric_name: str,
) -> dict[tuple[str, str], dict[int, float]]:
    if "env_name" not in dataset.runs.columns or "env_name" not in dataset.metrics.columns:
        raise ValueError("bakeoff datasets require an environment dimension")

    final_metrics = (
        dataset.metrics.filter(pl.col("metric") == metric_name)
        .sort(["run_id", "execution_id", "step"])
        .group_by(["run_id", "execution_id"])
        .agg(pl.col("value").last().alias("value"))
    )
    joined = final_metrics.join(
        dataset.runs.select(["run_id", "seed", "condition_name", "env_name"]),
        on="run_id",
        how="inner",
    ).filter(pl.col("condition_name").is_in(algorithms) & pl.col("env_name").is_in(environments))

    data_by_group: dict[tuple[str, str], dict[int, float]] = {}
    for row in joined.iter_rows(named=True):
        key = (str(row["condition_name"]), str(row["env_name"]))
        data_by_group.setdefault(key, {})[int(row["seed"])] = float(row["value"])
    return data_by_group


@dataclass(frozen=True)
class _BakeoffStatistics:
    normalized_scores: dict[str, dict[str, float]]
    performance_profiles: dict[str, dict[float, float]]
    probability_of_improvement: dict[str, dict[str, float | None]]
    optimality_gaps: dict[str, float]
    dispersion: dict[str, float]
    lower_tail: dict[str, float]
    supplementary_iqm: dict[str, float]
    omnibus_details: StatisticalTestDetails
    posthoc_matrix: dict[str, dict[str, float | None]]
    score_curves: dict[str, tuple[list[float], list[float]]]


def _compute_bakeoff_statistics(
    data_by_group: dict[tuple[str, str], dict[int, float]],
    algorithms: list[str],
    environments: list[str],
    references: dict[str, ReferenceRange],
    spec: ComparisonSpec,
) -> _BakeoffStatistics:
    normalized: dict[tuple[str, str], dict[int, float]] = {}
    for (algo, env), seed_map in data_by_group.items():
        if env not in references:
            raise ValueError(f"missing fixed reference for environment {env!r}")
        values = normalize_scores(np.array(list(seed_map.values())), references[env])
        for seed in seed_map:
            normalized.setdefault((algo, env), {})[seed] = float(values[list(seed_map).index(seed)])

    for env in environments:
        if not any(normalized.get((algo, env)) for algo in algorithms):
            raise ValueError(f"bakeoff environment {env!r} has no observed data")

    estimator = np.median if spec.estimand == "median" else np.mean
    normalized_means: dict[str, dict[str, float]] = {algo: {} for algo in algorithms}
    for algo in algorithms:
        for env in environments:
            values = normalized.get((algo, env), {}).values()
            normalized_means[algo][env] = float(estimator(list(values))) if values else float("nan")

    row_keys = []
    for env in environments:
        all_seeds_for_env = set()
        for algo in algorithms:
            all_seeds_for_env.update(normalized.get((algo, env), {}).keys())
        for seed in sorted(all_seeds_for_env):
            row_keys.append((env, seed))

    matrix = []
    for env, seed in row_keys:
        row = []
        for algo in algorithms:
            val = normalized.get((algo, env), {}).get(seed)
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix, dtype=np.float64)
    observed_counts = np.isfinite(matrix).sum(axis=1)
    valid_rows = observed_counts >= 2
    dropped_singleton_blocks = int(np.sum(observed_counts == 1))
    matrix = matrix[valid_rows]
    if matrix.shape[0] < spec.min_samples:
        raise ValueError(
            f"need at least {spec.min_samples} blocks with at least two observed algorithms; "
            f"got {matrix.shape[0]}"
        )
    has_missing = bool(np.isnan(matrix).any())
    if has_missing and spec.missingness == "error":
        raise ValueError("bakeoff contains missing cells under missingness='error'")
    rng = np.random.default_rng(spec.random_seed)
    block_result = block_stratified_permutation_test(
        matrix,
        n_resamples=spec.n_resamples,
        min_blocks=spec.min_samples,
        rng=rng,
    )

    raw_posthoc: dict[tuple[str, str], float] = {}
    pair_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for i, algo_a in enumerate(algorithms):
        for j, algo_b in enumerate(algorithms):
            if i >= j:
                continue
            paired_a = []
            paired_b = []
            for env, seed in row_keys:
                va = normalized.get((algo_a, env), {}).get(seed)
                vb = normalized.get((algo_b, env), {}).get(seed)
                if va is not None and vb is not None:
                    paired_a.append(va)
                    paired_b.append(vb)
            values_a = np.array(paired_a)
            values_b = np.array(paired_b)
            pair_data[(algo_a, algo_b)] = (values_a, values_b)
            pair_data[(algo_b, algo_a)] = (values_b, values_a)
            if len(paired_a) < spec.min_samples:
                continue
            result = run_permutation_test(
                values_a,
                values_b,
                estimand=spec.estimand,
                pairing="paired",
                alpha=spec.alpha,
                n_resamples=spec.n_resamples,
                min_samples=spec.min_samples,
                rng=rng,
            )
            raw_posthoc[(algo_a, algo_b)] = result.p_value

    adjusted_values = holm_step_down(list(raw_posthoc.values()))
    posthoc = {
        algo: {other: 1.0 if algo == other else None for other in algorithms}
        for algo in algorithms
    }
    for (algo_a, algo_b), adjusted in zip(raw_posthoc, adjusted_values, strict=True):
        posthoc[algo_a][algo_b] = adjusted
        posthoc[algo_b][algo_a] = adjusted

    threshold_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    profiles: dict[str, dict[float, float]] = {}
    gaps: dict[str, float] = {}
    dispersions: dict[str, float] = {}
    lower_tails: dict[str, float] = {}
    iqms: dict[str, float] = {}
    score_curves = {}
    for algo in algorithms:
        all_vals = []
        for env in environments:
            all_vals.extend(normalized.get((algo, env), {}).values())
        if all_vals:
            sorted_vals = sorted(all_vals)
            proportions = [i / len(sorted_vals) for i in range(1, len(sorted_vals) + 1)]
            score_curves[algo] = (sorted_vals, proportions)
            profiles[algo] = performance_profile(
                np.array(all_vals)[:, np.newaxis], threshold_values, [algo]
            )[algo]
            gaps[algo] = optimality_gap(np.array(all_vals))
            dispersions[algo] = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0
            lower_tails[algo] = float(np.quantile(all_vals, 0.1))
            iqms[algo] = interquartile_mean(np.array(all_vals))

    probabilities: dict[str, dict[str, float | None]] = {algo: {} for algo in algorithms}
    for algo_a in algorithms:
        for algo_b in algorithms:
            if algo_a == algo_b:
                continue
            values_a, values_b = pair_data[(algo_a, algo_b)]
            probabilities[algo_a][algo_b] = (
                probability_of_improvement(values_a, values_b) if len(values_a) else None
            )

    return _BakeoffStatistics(
        normalized_scores=normalized_means,
        performance_profiles=profiles,
        probability_of_improvement=probabilities,
        optimality_gaps=gaps,
        dispersion=dispersions,
        lower_tail=lower_tails,
        supplementary_iqm=iqms,
        omnibus_details=StatisticalTestDetails(
            test_name="Block-stratified permutation omnibus test",
            statistic=block_result.statistic,
            p_value=block_result.p_value,
            effect_size=None,
            is_significant=block_result.p_value < spec.alpha,
            assumptions={
                "has_missing": has_missing,
                "n_blocks": block_result.n_blocks,
                "n_observations": block_result.n_observations,
                "alpha": spec.alpha,
                "n_resamples": spec.n_resamples,
                "min_samples": spec.min_samples,
                "missingness": spec.missingness,
                "dropped_singleton_blocks": dropped_singleton_blocks,
            },
            justification="Algorithm labels were permuted within each observed block, retaining incomplete block patterns.",
            intermediate_tests={},
        ),
        posthoc_matrix=posthoc,
        score_curves=score_curves,
    )


def _plot_bakeoff(
    score_curves: dict[str, tuple[list[float], list[float]]],
    experiment_slug: str,
) -> Path | None:
    plot_path = Path("results/analysis") / experiment_slug / "normalized_score_comparison.png"
    if score_curves:
        plot_ecdf(score_curves, plot_path)
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

    table = Table(title="Reference-Normalized Performance Matrix", show_header=True, header_style="bold magenta")
    table.add_column("Algorithm")
    for env in environments:
        table.add_column(env, justify="right")
    for algo in report.algorithms:
        row = [algo]
        for env in environments:
            row.append(f"{report.normalized_scores[algo][env]:.2f}")
        table.add_row(*row)
    console.print(table)

    summary = Table(title="Aggregate Context", show_header=True, header_style="bold magenta")
    summary.add_column("Algorithm")
    summary.add_column("Dispersion", justify="right")
    summary.add_column("Lower tail (10%)", justify="right")
    summary.add_column("IQM (supplementary)", justify="right")
    for algo in report.algorithms:
        summary.add_row(
            algo,
            f"{report.dispersion[algo]:.2f}",
            f"{report.lower_tail[algo]:.2f}",
            f"{report.supplementary_iqm[algo]:.2f}",
        )
    console.print(summary)

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
                    "Scores are normalized against fixed caller-supplied references, so adding an algorithm "
                    "cannot change existing normalized values."
                ),
            ),
            title="Omnibus Methodology Justification",
        )
    )

    if report.aggregate_plot_path:
        console.print(
            Panel(
                Text(f"Saved normalized score visual artifact to: {report.aggregate_plot_path}", style="italic green"),
                title="Artifact Outputs Panel",
            )
        )


def compare_bakeoff(
    dataset: ExperimentDataset,
    algorithms: list[str],
    environments: list[str],
    metric_name: str,
    verbose: bool = True,
    references: dict[str, ReferenceRange] | None = None,
    spec: ComparisonSpec | None = None,
) -> BenchmarkBakeoffReport:
    """Compare algorithms with fixed-reference normalization and blockwise inference."""
    if len(algorithms) < 3:
        raise ValueError(
            f"compare_bakeoff requires at least 3 algorithms for the omnibus test, got "
            f"{len(algorithms)} ({algorithms!r}). For a two-condition comparison, use compare_pairwise instead."
        )
    if references is None:
        raise ValueError("compare_bakeoff requires fixed references for every environment")
    if spec is None:
        spec = ComparisonSpec(missingness="blockwise")
    elif spec.missingness == "complete_case":
        raise ValueError("bakeoff missingness must be 'blockwise' or 'error'")
    data_by_group = _load_bakeoff_data(
        dataset,
        algorithms,
        environments,
        metric_name,
    )

    statistics = _compute_bakeoff_statistics(
        data_by_group,
        algorithms,
        environments,
        references,
        spec,
    )
    normalized_scores = statistics.normalized_scores
    omnibus_details = statistics.omnibus_details
    posthoc = statistics.posthoc_matrix
    score_curves = statistics.score_curves

    plot_path = _plot_bakeoff(score_curves, dataset.experiment_name)

    report = BenchmarkBakeoffReport(
        algorithms=algorithms,
        environments=environments,
        normalized_scores=normalized_scores,
        performance_profiles=statistics.performance_profiles,
        probability_of_improvement=statistics.probability_of_improvement,
        optimality_gaps=statistics.optimality_gaps,
        dispersion=statistics.dispersion,
        lower_tail=statistics.lower_tail,
        supplementary_iqm=statistics.supplementary_iqm,
        omnibus_details=omnibus_details,
        posthoc_matrix=posthoc,
        aggregate_plot_path=plot_path,
    )

    if verbose:
        _present_bakeoff(report, environments, dataset.experiment_name)

    return report
