"""Private bakeoff reporting implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

from research_analysis._reporting_console import _present_bakeoff_report
from research_analysis._reporting_hypers import _load_dataset_hyperparameter_records, best_hypers
from research_analysis._reporting_models import (
    BakeoffReport,
    BenchmarkBakeoffReport,
    BlockKey,
    CellSummary,
    HyperparameterConfigReport,
    HyperparameterSliceReport,
    PairedDifference,
    StatisticalTestDetails,
)
from research_analysis._reporting_plots import _plot_run_distributions
from research_analysis._reporting_warnings import (
    boundary_warning,
    insufficient_seeds_warning,
    non_normality_warning,
    variance_warning,
)
from research_analysis.bootstrap import bootstrap_difference_ci, bootstrap_scalar_ci
from research_analysis.comparison import BakeoffSpec
from research_analysis.curves import flatline_flags
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.diagnostics.bakeoff import BakeoffWarningAccumulator, BakeoffWarningCode
from research_analysis.duplicates import duplicate_seed_error
from research_analysis.multiplicity import bonferroni_confidence, holm_step_down
from research_analysis.non_finite import require_finite
from research_analysis.performance import FINAL_VALUE, PerformanceMeasure, run_scores
from research_analysis.permutation import run_permutation_test
from research_analysis.repeated_blocks import block_stratified_permutation_test
from research_analysis.statistics import (
    minimum_tolerance_samples,
    pointwise_tolerance_interval,
    select_median_run_index,
)


def _block_label(block_key: BlockKey) -> str:
    return "|".join(str(value) for _column, value in block_key)


def _load_bakeoff_data(
    dataset: ExperimentDataset,
    conditions: list[str],
    block_by: list[str],
    metric_name: str,
    measure: PerformanceMeasure,
) -> tuple[dict[tuple[str, BlockKey], dict[int, float]], dict[tuple[str, BlockKey, int], int]]:
    missing = [column for column in block_by if column not in dataset.runs.columns]
    if missing:
        raise ValueError(f"block_by columns not present in the dataset: {missing}")

    scores = run_scores(dataset.metrics, metric_name, measure).select(["run_id", "value"])
    joined = scores.join(
        dataset.runs.select(["run_id", "seed", "condition_name", *block_by]),
        on="run_id",
        how="inner",
    ).filter(pl.col("condition_name").is_in(conditions))
    require_finite(joined, f"{metric_name!r} run scores")

    data_by_group: dict[tuple[str, BlockKey], dict[int, float]] = {}
    run_ids: dict[tuple[str, BlockKey, int], int] = {}
    for row in joined.iter_rows(named=True):
        block_key: BlockKey = tuple((column, row[column]) for column in block_by)
        condition, seed = str(row["condition_name"]), int(row["seed"])
        if (condition, block_key, seed) in run_ids:
            cell = f"condition {condition!r} in block {_block_label(block_key)!r}"
            raise duplicate_seed_error(cell, seed, (run_ids[(condition, block_key, seed)], row["run_id"]))
        run_ids[(condition, block_key, seed)] = int(row["run_id"])
        data_by_group.setdefault((condition, block_key), {})[seed] = float(row["value"])
    return data_by_group, run_ids


@dataclass(frozen=True)
class _BakeoffStatistics:
    omnibus_details: StatisticalTestDetails
    differences: tuple[PairedDifference, ...]


def _paired_differences(
    scores: dict[tuple[str, BlockKey], dict[int, float]],
    conditions: list[str],
    block_keys: list[BlockKey],
    spec: BakeoffSpec,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[PairedDifference, ...]:
    """Compare every pair of conditions within each block, never pooling runs across blocks."""
    planned: list[PairedDifference] = []
    for block_key in block_keys:
        for index, minuend in enumerate(conditions):
            for subtrahend in conditions[index + 1 :]:
                planned.append(_paired_difference(scores, minuend, subtrahend, block_key, spec, confidence, rng))

    # A skipped pair still belongs to the family: counting it as p=1 keeps Holm's divisor honest.
    adjusted = holm_step_down([1.0 if item.p_value is None else item.p_value for item in planned])
    return tuple(
        item if item.p_value is None else replace(item, p_value=p_value)
        for item, p_value in zip(planned, adjusted, strict=True)
    )


def _paired_difference(
    scores: dict[tuple[str, BlockKey], dict[int, float]],
    minuend: str,
    subtrahend: str,
    block_key: BlockKey,
    spec: BakeoffSpec,
    confidence: float,
    rng: np.random.Generator,
) -> PairedDifference:
    runs_a = scores.get((minuend, block_key), {})
    runs_b = scores.get((subtrahend, block_key), {})
    seeds = sorted(set(runs_a) & set(runs_b))
    skipped = PairedDifference(
        minuend=minuend,
        subtrahend=subtrahend,
        block=_block_label(block_key),
        estimate=None,
        ci=None,
        confidence=confidence,
        n_paired=len(seeds),
        method="paired percentile bootstrap",
        p_value=None,
        skipped_reason=f"only {len(seeds)} paired runs, fewer than min_samples={spec.min_samples}",
    )
    if len(seeds) < spec.min_samples:
        return skipped

    values_a = np.array([runs_a[seed] for seed in seeds])
    values_b = np.array([runs_b[seed] for seed in seeds])
    test = run_permutation_test(
        values_a,
        values_b,
        estimand=spec.estimand,
        pairing="paired",
        alpha=spec.alpha,
        n_resamples=spec.n_resamples,
        min_samples=spec.min_samples,
        rng=rng,
    )
    interval = bootstrap_difference_ci(
        values_a,
        values_b,
        estimand=spec.estimand,
        pairing="paired",
        confidence=confidence,
        n_resamples=spec.n_resamples,
        min_samples=spec.min_samples,
        rng=rng,
    )
    return replace(
        skipped,
        estimate=interval.estimate,
        ci=(interval.ci_low, interval.ci_high),
        p_value=test.p_value,
        skipped_reason=None,
    )


def _compute_bakeoff_statistics(
    data_by_group: dict[tuple[str, BlockKey], dict[int, float]],
    conditions: list[str],
    block_keys: list[BlockKey],
    spec: BakeoffSpec,
    comparison_confidence: float,
) -> _BakeoffStatistics:
    for block_key in block_keys:
        if not any(data_by_group.get((condition, block_key)) for condition in conditions):
            raise ValueError(f"bakeoff block {_block_label(block_key)!r} has no observed data")

    row_keys = []
    for block_key in block_keys:
        all_seeds_for_block = set()
        for condition in conditions:
            all_seeds_for_block.update(data_by_group.get((condition, block_key), {}).keys())
        for seed in sorted(all_seeds_for_block):
            row_keys.append((block_key, seed))

    matrix = []
    for block_key, seed in row_keys:
        row = []
        for condition in conditions:
            val = data_by_group.get((condition, block_key), {}).get(seed)
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

    differences = _paired_differences(data_by_group, conditions, block_keys, spec, comparison_confidence, rng)

    return _BakeoffStatistics(
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
        differences=differences,
    )


def _summarize_cell(
    values: list[float], run_ids: list[int], spec: BakeoffSpec, rng: np.random.Generator
) -> CellSummary:
    array = np.asarray(values, dtype=np.float64)
    min_runs = minimum_tolerance_samples(confidence=spec.tolerance_confidence, coverage=spec.tolerance_coverage)
    tolerance = None
    if array.size >= min_runs:
        interval = pointwise_tolerance_interval(
            array, confidence=spec.tolerance_confidence, coverage=spec.tolerance_coverage
        )
        tolerance = (float(interval.ci_low), float(interval.ci_high))
    summary = CellSummary(
        n=int(array.size),
        estimate=None,
        ci=None,
        confidence=spec.confidence_level,
        sd=None,
        tolerance=tolerance,
        tolerance_min_runs=min_runs,
        median_run_id=None,
    )
    if array.size == 0:
        return summary

    summary = replace(
        summary,
        estimate=float(np.median(array) if spec.estimand == "median" else np.mean(array)),
        median_run_id=run_ids[select_median_run_index(array)],
    )
    if array.size < 2:
        return summary

    ci = bootstrap_scalar_ci(
        array,
        estimand=spec.estimand,
        confidence=spec.confidence_level,
        n_resamples=spec.n_resamples,
        min_samples=2,
        rng=rng,
    )
    return replace(summary, ci=(ci.ci_low, ci.ci_high), sd=float(np.std(array, ddof=1)))


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
    spec: BakeoffSpec | None = None,
    higher_is_better: bool = True,
    min_seeds: int = 5,
    verbose: bool = True,
    measure: PerformanceMeasure = FINAL_VALUE,
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
            n_resamples=spec.n_resamples,
            min_samples=spec.min_samples,
            random_seed=spec.random_seed,
            measure=measure,
        )
        block_champions = {key[0]: report for key, report in raw_champions.items() if key[0] in conditions}
        champions[_block_label(block_key)] = block_champions

        for axis in select_champion_over:
            # Raw per-value means, not best_hypers/slice_hypers: those require at least one
            # OTHER declared axis to search over while vary is grouped out, which a
            # single-axis sweep (very common) can never satisfy.
            records = _load_dataset_hyperparameter_records(
                block_dataset, [dataset.dimensions.condition, axis], metric_name, measure
            )
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
    data_by_group, run_ids = _load_bakeoff_data(combined, conditions, block_by, metric_name, measure)
    family_size = len(block_keys) * len(conditions) * (len(conditions) - 1) // 2
    comparison_confidence = bonferroni_confidence(spec.confidence_level, family_size)
    comparison_statistics = _compute_bakeoff_statistics(
        data_by_group, conditions, block_keys, spec, comparison_confidence
    )

    cells: dict[str, dict[str, CellSummary]] = {condition: {} for condition in conditions}
    cell_rng = np.random.default_rng(spec.random_seed)

    for block_key in block_keys:
        label = _block_label(block_key)
        for condition in conditions:
            seed_map = data_by_group.get((condition, block_key), {})
            values = list(seed_map.values())
            cell_run_ids = [run_ids[(condition, block_key, seed)] for seed in seed_map]
            cells[condition][label] = _summarize_cell(values, cell_run_ids, spec, cell_rng)

            if not values:
                continue
            variance_issue = variance_warning(values, condition=condition, cell=block_key)
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

        block_metrics = _filter_dataset(combined, _block_predicate(block_key)).metrics
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
        measure=measure,
        spec=spec,
        cells=cells,
        champions=champions or None,
        comparison=BenchmarkBakeoffReport(
            algorithms=conditions,
            environments=[_block_label(block_key) for block_key in block_keys],
            omnibus_details=comparison_statistics.omnibus_details,
        ),
        differences=comparison_statistics.differences,
        distribution_plot_paths=_plot_run_distributions(
            {
                _block_label(block_key): {
                    condition: list(seed_map.values())
                    for condition in conditions
                    if (seed_map := data_by_group.get((condition, block_key)))
                }
                for block_key in block_keys
            },
            metric_name,
            dataset.experiment_name,
        ),
        family_size=family_size,
        comparison_confidence=comparison_confidence,
        warnings=warnings.as_tuple(),
    )

    if verbose:
        _present_bakeoff_report(report, dataset.experiment_name)

    return report
