"""Automated statistical reporting API for RL experiments.

Implements pairwise comparisons, hyperparameter sensitivity analysis, and benchmark bakeoffs
with pedagogical explanations and diagnostic plotting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from research_analysis._reporting_console import (
    _present_bakeoff_report,
)
from research_analysis._reporting_hypers import (
    _load_dataset_hyperparameter_records as _load_dataset_hyperparameter_records,
)
from research_analysis._reporting_hypers import (
    best_hypers as best_hypers,
)
from research_analysis._reporting_hypers import (
    slice_hypers as slice_hypers,
)
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
from research_analysis._reporting_plots import _plot_bakeoff
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

# ── Entry Points ─────────────────────────────────────────────────────────────


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
