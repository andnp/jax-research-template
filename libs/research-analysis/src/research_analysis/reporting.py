"""Automated statistical reporting API for RL experiments.

Implements pairwise comparisons, hyperparameter sensitivity analysis, and benchmark bakeoffs
with pedagogical explanations and diagnostic plotting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

import numpy as np
import polars as pl
from research_plot.plots import plot_ecdf, plot_sensitivity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scipy import stats as scipy_stats

from research_analysis._reporting_models import (
    BakeoffReport as BakeoffReport,
)
from research_analysis._reporting_models import (
    BenchmarkBakeoffReport as BenchmarkBakeoffReport,
)
from research_analysis._reporting_models import (
    BlockKey as BlockKey,
)
from research_analysis._reporting_models import (
    HyperparameterConfigReport as HyperparameterConfigReport,
)
from research_analysis._reporting_models import (
    HyperparameterDifferenceReport as HyperparameterDifferenceReport,
)
from research_analysis._reporting_models import (
    HyperparameterSliceReport as HyperparameterSliceReport,
)
from research_analysis._reporting_models import (
    StatisticalTestDetails as StatisticalTestDetails,
)
from research_analysis.aggregates import (
    ReferenceRange,
    interquartile_mean,
    normalize_scores,
    optimality_gap,
    performance_profile,
    probability_of_improvement,
)
from research_analysis.bootstrap import bootstrap_difference_ci
from research_analysis.comparison import BakeoffSpec
from research_analysis.curves import flatline_flags
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.diagnostics.bakeoff import BakeoffWarning, BakeoffWarningAccumulator, BakeoffWarningCode
from research_analysis.multiplicity import holm_step_down
from research_analysis.permutation import run_permutation_test
from research_analysis.repeated_blocks import block_stratified_permutation_test
from research_analysis.tuning import _build_hyper_matrix, hierarchical_bootstrap_ci, hierarchical_bootstrap_difference_ci

# ── Entry Points ─────────────────────────────────────────────────────────────


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
    n_candidates = matrix.shape[0]
    normalized_entropy = float(bootstrap.selection_entropy / np.log(n_candidates)) if n_candidates > 1 else 0.0

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
        normalized_selection_entropy=normalized_entropy,
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


def non_normality_warning(
    values: Sequence[float] | np.ndarray,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    alpha: float = 0.05,
) -> BakeoffWarning | None:
    """Shapiro-Wilk normality check against one cell's raw values, informational only.

    Per ADR-010, no normality pretest may select which statistical test runs; this
    never gates anything, it only surfaces a caveat for the reader to weigh (e.g.
    whether a median estimand would be more appropriate than a mean).
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size < 3:
        return None
    _statistic, p_value = scipy_stats.shapiro(array)
    if p_value >= alpha:
        return None
    return BakeoffWarning(
        code=BakeoffWarningCode.NON_NORMAL_DISTRIBUTION,
        message=f"Shapiro-Wilk rejects normality (p={p_value:.4f} < alpha={alpha}); consider a robust estimand.",
        condition=condition,
        cell=cell,
    )


def insufficient_seeds_warning(
    report: HyperparameterConfigReport,
    n_seeds: int,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    min_seeds: int = 5,
    max_normalized_entropy: float = 0.5,
) -> BakeoffWarning | None:
    """Flag a champion pick backed by too few seeds, or unstable under resampling.

    Fires on either signal: fewer than ``min_seeds`` raw seeds, or a
    ``normalized_selection_entropy`` above ``max_normalized_entropy`` -- a low seed
    count can still bootstrap a stable pick, and a high entropy can occur even with
    many seeds when several configurations perform near-identically.
    """
    if n_seeds < min_seeds:
        return BakeoffWarning(
            code=BakeoffWarningCode.INSUFFICIENT_SEEDS,
            message=f"only {n_seeds} seed(s) backing this champion, fewer than the recommended {min_seeds}.",
            condition=condition,
            cell=cell,
        )
    if report.normalized_selection_entropy > max_normalized_entropy:
        return BakeoffWarning(
            code=BakeoffWarningCode.INSUFFICIENT_SEEDS,
            message=(
                f"champion selection is unstable under resampling (normalized entropy "
                f"{report.normalized_selection_entropy:.2f} > {max_normalized_entropy}); "
                "several configurations perform near-identically."
            ),
            condition=condition,
            cell=cell,
        )
    return None


def variance_warning(
    values: Sequence[float] | np.ndarray,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    reference_range: float | None = None,
    extreme_relative_std: float = 1.0,
) -> BakeoffWarning | None:
    """Flag a cell with zero cross-seed variance, or variance extreme relative to the
    metric's own cross-condition range (``reference_range``, typically max minus min
    across every condition and cell in the same report).
    """
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if std == 0.0:
        return BakeoffWarning(
            code=BakeoffWarningCode.ZERO_VARIANCE,
            message="every seed produced an identical value; check for a fixed seed or a saturated metric.",
            condition=condition,
            cell=cell,
        )
    if reference_range is not None and reference_range > 0 and std > extreme_relative_std * reference_range:
        return BakeoffWarning(
            code=BakeoffWarningCode.EXTREME_VARIANCE,
            message=(
                f"cross-seed std ({std:.4g}) exceeds {extreme_relative_std:.0%} of the metric's own "
                f"cross-condition range ({reference_range:.4g})."
            ),
            condition=condition,
            cell=cell,
        )
    return None


def _block_label(block_key: BlockKey) -> str:
    return "|".join(str(value) for _column, value in block_key)


def _load_bakeoff_data(
    dataset: ExperimentDataset,
    conditions: list[str],
    block_by: list[str],
    metric_name: str,
) -> dict[tuple[str, BlockKey], dict[int, float]]:
    missing = [column for column in block_by if column not in dataset.runs.columns]
    if missing:
        raise ValueError(f"block_by columns not present in the dataset: {missing}")

    final_metrics = (
        dataset.metrics.filter(pl.col("metric") == metric_name)
        .sort(["run_id", "execution_id", "step"])
        .group_by(["run_id", "execution_id"])
        .agg(pl.col("value").last().alias("value"))
    )
    joined = final_metrics.join(
        dataset.runs.select(["run_id", "seed", "condition_name", *block_by]),
        on="run_id",
        how="inner",
    ).filter(pl.col("condition_name").is_in(conditions))

    data_by_group: dict[tuple[str, BlockKey], dict[int, float]] = {}
    for row in joined.iter_rows(named=True):
        block_key: BlockKey = tuple((column, row[column]) for column in block_by)
        key = (str(row["condition_name"]), block_key)
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
    difference_estimate: dict[str, dict[str, float | None]]
    difference_ci: dict[str, dict[str, tuple[float, float] | None]]
    score_curves: dict[str, tuple[list[float], list[float]]]


def _compute_bakeoff_statistics(
    data_by_group: dict[tuple[str, BlockKey], dict[int, float]],
    conditions: list[str],
    block_keys: list[BlockKey],
    references: dict[str, ReferenceRange] | None,
    spec: BakeoffSpec,
) -> _BakeoffStatistics:
    if references is None and len(block_keys) > 1:
        raise ValueError(
            "pooling raw (unnormalized) scores across more than one block requires "
            "references; supply references or narrow block_by to a single block"
        )

    if references is not None:
        normalized: dict[tuple[str, BlockKey], dict[int, float]] = {}
        for (condition, block_key), seed_map in data_by_group.items():
            label = _block_label(block_key)
            if label not in references:
                raise ValueError(f"missing fixed reference for block {label!r}")
            values = normalize_scores(np.array(list(seed_map.values())), references[label])
            for seed, value in zip(seed_map, values, strict=True):
                normalized.setdefault((condition, block_key), {})[seed] = float(value)
    else:
        normalized = data_by_group

    for block_key in block_keys:
        if not any(normalized.get((condition, block_key)) for condition in conditions):
            raise ValueError(f"bakeoff block {_block_label(block_key)!r} has no observed data")

    estimator = np.median if spec.estimand == "median" else np.mean
    normalized_means: dict[str, dict[str, float]] = {condition: {} for condition in conditions}
    for condition in conditions:
        for block_key in block_keys:
            values = normalized.get((condition, block_key), {}).values()
            normalized_means[condition][_block_label(block_key)] = float(estimator(list(values))) if values else float("nan")

    row_keys = []
    for block_key in block_keys:
        all_seeds_for_block = set()
        for condition in conditions:
            all_seeds_for_block.update(normalized.get((condition, block_key), {}).keys())
        for seed in sorted(all_seeds_for_block):
            row_keys.append((block_key, seed))

    matrix = []
    for block_key, seed in row_keys:
        row = []
        for condition in conditions:
            val = normalized.get((condition, block_key), {}).get(seed)
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix, dtype=np.float64)
    observed_counts = np.isfinite(matrix).sum(axis=1)
    incomplete_rows = int(np.sum((observed_counts >= 1) & (observed_counts < len(conditions))))
    if incomplete_rows > 0 and spec.missingness == "error":
        raise ValueError(f"bakeoff contains {incomplete_rows} incomplete block(s) under missingness='error'")
    # complete_case keeps only rows where every condition was observed; blockwise keeps
    # any row with at least two observed conditions, retaining usable incomplete blocks.
    required_observations = len(conditions) if spec.missingness == "complete_case" else 2
    valid_rows = observed_counts >= required_observations
    dropped_singleton_blocks = int(np.sum((observed_counts >= 1) & (observed_counts < required_observations)))
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
    raw_difference: dict[tuple[str, str], float] = {}
    raw_difference_ci: dict[tuple[str, str], tuple[float, float]] = {}
    pair_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for i, condition_a in enumerate(conditions):
        for j, condition_b in enumerate(conditions):
            if i >= j:
                continue
            paired_a = []
            paired_b = []
            for block_key, seed in row_keys:
                va = normalized.get((condition_a, block_key), {}).get(seed)
                vb = normalized.get((condition_b, block_key), {}).get(seed)
                if va is not None and vb is not None:
                    paired_a.append(va)
                    paired_b.append(vb)
            values_a = np.array(paired_a)
            values_b = np.array(paired_b)
            pair_data[(condition_a, condition_b)] = (values_a, values_b)
            pair_data[(condition_b, condition_a)] = (values_b, values_a)
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
            raw_posthoc[(condition_a, condition_b)] = result.p_value
            raw_difference[(condition_a, condition_b)] = result.statistic
            interval = bootstrap_difference_ci(
                values_a,
                values_b,
                estimand=spec.estimand,
                pairing="paired",
                confidence=spec.confidence_level,
                n_resamples=spec.n_resamples,
                min_samples=spec.min_samples,
                rng=rng,
            )
            raw_difference_ci[(condition_a, condition_b)] = (interval.ci_low, interval.ci_high)

    adjusted_values = holm_step_down(list(raw_posthoc.values()))
    posthoc = {
        condition: {other: 1.0 if condition == other else None for other in conditions}
        for condition in conditions
    }
    for (condition_a, condition_b), adjusted in zip(raw_posthoc, adjusted_values, strict=True):
        posthoc[condition_a][condition_b] = adjusted
        posthoc[condition_b][condition_a] = adjusted

    difference_estimate: dict[str, dict[str, float | None]] = {
        condition: {other: None for other in conditions} for condition in conditions
    }
    difference_ci: dict[str, dict[str, tuple[float, float] | None]] = {
        condition: {other: None for other in conditions} for condition in conditions
    }
    for (condition_a, condition_b), diff in raw_difference.items():
        ci_low, ci_high = raw_difference_ci[(condition_a, condition_b)]
        difference_estimate[condition_a][condition_b] = diff
        difference_estimate[condition_b][condition_a] = -diff
        difference_ci[condition_a][condition_b] = (ci_low, ci_high)
        difference_ci[condition_b][condition_a] = (-ci_high, -ci_low)

    threshold_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    profiles: dict[str, dict[float, float]] = {}
    gaps: dict[str, float] = {}
    dispersions: dict[str, float] = {}
    lower_tails: dict[str, float] = {}
    iqms: dict[str, float] = {}
    score_curves = {}
    for condition in conditions:
        all_vals = []
        for block_key in block_keys:
            all_vals.extend(normalized.get((condition, block_key), {}).values())
        if all_vals:
            sorted_vals = sorted(all_vals)
            proportions = [i / len(sorted_vals) for i in range(1, len(sorted_vals) + 1)]
            score_curves[condition] = (sorted_vals, proportions)
            if references is not None:
                # performance_profile/optimality_gap assume a normalized [0, 1]-ish
                # scale; without references there is no anchor to interpret them against.
                profiles[condition] = performance_profile(
                    np.array(all_vals)[:, np.newaxis], threshold_values, [condition]
                )[condition]
                gaps[condition] = optimality_gap(np.array(all_vals))
            dispersions[condition] = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0
            lower_tails[condition] = float(np.quantile(all_vals, 0.1))
            iqms[condition] = interquartile_mean(np.array(all_vals))

    probabilities: dict[str, dict[str, float | None]] = {condition: {} for condition in conditions}
    for condition_a in conditions:
        for condition_b in conditions:
            if condition_a == condition_b:
                continue
            values_a, values_b = pair_data[(condition_a, condition_b)]
            probabilities[condition_a][condition_b] = (
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
        difference_estimate=difference_estimate,
        difference_ci=difference_ci,
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


def _block_predicate(block_key: BlockKey) -> pl.Expr:
    predicate = pl.lit(True)
    for column, value in block_key:
        predicate = predicate & (pl.col(column) == value)
    return predicate


def _filter_dataset(dataset: ExperimentDataset, predicate: pl.Expr) -> ExperimentDataset:
    runs = dataset.runs.filter(predicate)
    metrics = dataset.metrics.join(runs.select("run_id"), on="run_id", how="inner")
    return replace(dataset, runs=runs, metrics=metrics)


def _filter_to_champions(dataset: ExperimentDataset, champions: dict[str, dict[str, Any]]) -> ExperimentDataset:
    """Restrict runs to each condition's champion hyperparameter combination."""
    condition_column = pl.col("condition_name")
    selected = pl.any_horizontal(
        *[
            condition_column.eq(condition) & pl.all_horizontal(*[pl.col(key) == value for key, value in config.items()])
            for condition, config in champions.items()
        ]
    )
    return _filter_dataset(dataset, selected)


def bakeoff_report(
    dataset: ExperimentDataset,
    metric_name: str,
    conditions: list[str],
    *,
    select_champion_over: list[str] | None = None,
    block_by: list[str] | None = None,
    references: dict[str, ReferenceRange] | None = None,
    spec: BakeoffSpec | None = None,
    higher_is_better: bool = True,
    min_seeds: int = 5,
    verbose: bool = True,
) -> BakeoffReport:
    """Champion selection, comparison, and statistical caveats, in one repeatable call.

    Compare two or more conditions using paired block-and-seed observations, with an
    optional champion-selection stage and warnings for unstable or poorly supported
    comparisons.
    """
    if len(conditions) < 2:
        raise ValueError(f"bakeoff_report requires at least 2 conditions, got {len(conditions)} ({conditions!r})")
    block_by = block_by or []
    if spec is None:
        spec = BakeoffSpec(missingness="blockwise" if block_by else "error")
    missing_block_columns = [column for column in block_by if column not in dataset.runs.columns]
    if missing_block_columns:
        raise ValueError(f"block_by columns not present in the dataset: {missing_block_columns}")
    declared_axes = {axis.name for axis in dataset.dimensions.sweep_axes}
    if select_champion_over is not None and set(select_champion_over) != declared_axes:
        raise ValueError(
            f"select_champion_over must name exactly the dataset's declared sweep axes "
            f"{sorted(declared_axes)}, got {sorted(select_champion_over)}"
        )

    block_keys: list[BlockKey] = (
        [tuple((column, row[column]) for column in block_by) for row in dataset.runs.select(block_by).unique().sort(block_by).iter_rows(named=True)]
        if block_by
        else [()]
    )

    warnings = BakeoffWarningAccumulator()
    champions: dict[str, dict[str, HyperparameterConfigReport]] = {}
    comparison_runs: list[pl.DataFrame] = []
    comparison_metrics: list[pl.DataFrame] = []

    for block_key in block_keys:
        block_dataset = _filter_dataset(dataset, _block_predicate(block_key)) if block_key else dataset
        cell = block_key

        if select_champion_over is None:
            comparison_runs.append(block_dataset.runs)
            comparison_metrics.append(block_dataset.metrics)
            continue

        raw_champions = best_hypers(
            block_dataset,
            metric_name,
            group_by=[dataset.dimensions.condition],
            higher_is_better=higher_is_better,
            verbose=False,
            confidence_level=spec.confidence_level,
        )
        block_champions = {key[0]: report for key, report in raw_champions.items() if key[0] in conditions}
        champions[_block_label(block_key)] = block_champions

        for axis in select_champion_over:
            # Raw per-value means, not best_hypers/slice_hypers: those require at least one
            # OTHER declared axis to search over while vary is grouped out, which a
            # single-axis sweep (very common) can never satisfy.
            records = _load_dataset_hyperparameter_records(block_dataset, [dataset.dimensions.condition, axis], metric_name)
            curves: dict[str, dict[Any, list[float]]] = {}
            for record in records:
                cond = record["hypers"][dataset.dimensions.condition]
                if cond not in conditions:
                    continue
                curves.setdefault(cond, {}).setdefault(record["hypers"][axis], []).append(record["value"])
            for cond, value_map in curves.items():
                sorted_values = sorted(value_map)
                means = [float(np.mean(value_map[value])) for value in sorted_values]
                slice_report = HyperparameterSliceReport(
                    vary=axis, values=sorted_values, best_performance=means, slice_performance=means, plot_path=None
                )
                warning = boundary_warning(slice_report, condition=cond, cell=cell, higher_is_better=higher_is_better)
                if warning is not None:
                    warnings.add(warning.code, warning.message, warning.condition, warning.cell)

        champion_dataset = _filter_to_champions(
            block_dataset, {condition: report.winning_config for condition, report in block_champions.items()}
        )
        comparison_runs.append(champion_dataset.runs)
        comparison_metrics.append(champion_dataset.metrics)

    combined = replace(dataset, runs=pl.concat(comparison_runs), metrics=pl.concat(comparison_metrics))
    data_by_group = _load_bakeoff_data(combined, conditions, block_by, metric_name)
    comparison_statistics = _compute_bakeoff_statistics(data_by_group, conditions, block_keys, references, spec)

    seed_counts: dict[str, dict[str, int]] = {condition: {} for condition in conditions}
    cell_means: dict[str, dict[str, float]] = {condition: {} for condition in conditions}
    cell_variances: dict[str, dict[str, float]] = {condition: {} for condition in conditions}
    all_values = [value for seed_map in data_by_group.values() for value in seed_map.values()]
    cross_condition_range = float(max(all_values) - min(all_values)) if all_values else 0.0

    for block_key in block_keys:
        label = _block_label(block_key)
        for condition in conditions:
            seed_map = data_by_group.get((condition, block_key), {})
            values = list(seed_map.values())
            seed_counts[condition][label] = len(values)
            cell_means[condition][label] = float(np.mean(values)) if values else float("nan")
            cell_variances[condition][label] = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0

            if not values:
                continue
            variance_issue = variance_warning(values, condition=condition, cell=block_key, reference_range=cross_condition_range)
            if variance_issue is not None:
                warnings.add(variance_issue.code, variance_issue.message, variance_issue.condition, variance_issue.cell)
            if variance_issue is None or variance_issue.code != BakeoffWarningCode.ZERO_VARIANCE:
                # Shapiro-Wilk is undefined (and noisy) on constant data; zero variance
                # is already reported above, so skip the redundant, degenerate check.
                normality_issue = non_normality_warning(values, condition=condition, cell=block_key)
                if normality_issue is not None:
                    warnings.add(normality_issue.code, normality_issue.message, normality_issue.condition, normality_issue.cell)
            if condition in champions.get(label, {}):
                seeds_issue = insufficient_seeds_warning(
                    champions[label][condition], len(values), condition=condition, cell=block_key, min_seeds=min_seeds
                )
                if seeds_issue is not None:
                    warnings.add(seeds_issue.code, seeds_issue.message, seeds_issue.condition, seeds_issue.cell)

        block_metrics = combined.metrics.filter(_block_predicate(block_key)) if block_key else combined.metrics
        flat = flatline_flags(block_metrics, metric_name)
        for row in flat.filter(pl.col("flatlined")).iter_rows(named=True):
            if row["condition_name"] not in conditions:
                continue
            warnings.add(
                BakeoffWarningCode.FLATLINED_LEARNING_CURVE,
                f"mean curve stops moving well before the run ends (tail range "
                f"{row['tail_range']:.4g} of whole range {row['whole_range']:.4g}).",
                row["condition_name"],
                block_key,
            )

    report = BakeoffReport(
        conditions=conditions,
        metric_name=metric_name,
        seed_counts=seed_counts,
        cell_means=cell_means,
        cell_variances=cell_variances,
        champions=champions or None,
        comparison=BenchmarkBakeoffReport(
            algorithms=conditions,
            environments=[_block_label(block_key) for block_key in block_keys],
            normalized_scores=comparison_statistics.normalized_scores,
            performance_profiles=comparison_statistics.performance_profiles,
            probability_of_improvement=comparison_statistics.probability_of_improvement,
            optimality_gaps=comparison_statistics.optimality_gaps,
            dispersion=comparison_statistics.dispersion,
            lower_tail=comparison_statistics.lower_tail,
            supplementary_iqm=comparison_statistics.supplementary_iqm,
            omnibus_details=comparison_statistics.omnibus_details,
            posthoc_matrix=comparison_statistics.posthoc_matrix,
            difference_estimate=comparison_statistics.difference_estimate,
            difference_ci=comparison_statistics.difference_ci,
            aggregate_plot_path=_plot_bakeoff(comparison_statistics.score_curves, dataset.experiment_name),
        ),
        warnings=warnings.as_tuple(),
    )

    if verbose:
        _present_bakeoff_report(report, dataset.experiment_name)

    return report


def _present_bakeoff_report(report: BakeoffReport, experiment_slug: str) -> None:
    console = Console()
    console.print()
    console.print(Panel(Text(f"🏆 BAKEOFF REPORT: {report.conditions}", style="bold cyan"), subtitle=f"Experiment: {experiment_slug}"))

    blocks = sorted({label for per_condition in report.seed_counts.values() for label in per_condition})
    table = Table(title=f"Cells ({report.metric_name})", show_header=True, header_style="bold magenta")
    table.add_column("Condition")
    for label in blocks:
        table.add_column(label or "(no blocking)", justify="right")
    for condition in report.conditions:
        row = [condition]
        for label in blocks:
            n = report.seed_counts[condition].get(label, 0)
            mean = report.cell_means[condition].get(label, float("nan"))
            var = report.cell_variances[condition].get(label, float("nan"))
            row.append(f"{mean:.3g} ± {var:.2g} (n={n})")
        table.add_row(*row)
    console.print(table)

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

    diff_table = Table(title="Pairwise differences", show_header=True, header_style="bold magenta")
    diff_table.add_column("A")
    diff_table.add_column("B")
    diff_table.add_column("A − B")
    diff_table.add_column("95% CI")
    diff_table.add_column("Holm p")
    seen: set[tuple[str, str]] = set()
    for condition_a in report.conditions:
        for condition_b in report.conditions:
            if condition_a == condition_b or (condition_b, condition_a) in seen:
                continue
            seen.add((condition_a, condition_b))
            diff = report.comparison.difference_estimate[condition_a][condition_b]
            ci = report.comparison.difference_ci[condition_a][condition_b]
            p_value = report.comparison.posthoc_matrix[condition_a][condition_b]
            if diff is None or ci is None:
                continue
            diff_table.add_row(condition_a, condition_b, f"{diff:+.3g}", f"[{ci[0]:+.3g}, {ci[1]:+.3g}]", f"{p_value:.4f}" if p_value is not None else "—")
    console.print(diff_table)
