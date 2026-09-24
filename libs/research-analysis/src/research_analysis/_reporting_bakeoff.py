"""Private bakeoff reporting implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl

from research_analysis._reporting_hypers import (
    _best_hypers_difference,
    _load_dataset_hyperparameter_records,
    best_hypers,
)
from research_analysis._reporting_models import (
    BakeoffReport,
    BlockKey,
    CellSummary,
    HyperparameterConfigReport,
    HyperparameterSliceReport,
    PairedDifference,
    RunScore,
    StatisticalTestDetails,
)
from research_analysis._reporting_warnings import (
    boundary_warning,
    insufficient_seeds_warning,
    non_normality_warning,
    small_sample_coverage_warning,
    variance_warning,
)
from research_analysis.bootstrap import bootstrap_difference_ci, bootstrap_scalar_ci
from research_analysis.comparison import BakeoffSpec
from research_analysis.curves import flatline_flags
from research_analysis.dataset.types import ExperimentDataset, RunAccounting, SweepValueType
from research_analysis.diagnostics.bakeoff import (
    BakeoffWarning,
    BakeoffWarningCode,
    IncompleteBlocksError,
    IncompleteCell,
)
from research_analysis.diagnostics.types import InsufficientDataCell, InsufficientDataError
from research_analysis.duplicates import duplicate_seed_error
from research_analysis.multiplicity import bonferroni_confidence, holm_step_down
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
    block_datasets: list[tuple[BlockKey, ExperimentDataset]],
    metric_name: str,
    measure: PerformanceMeasure,
) -> tuple[dict[tuple[str, BlockKey], dict[int, float]], dict[tuple[str, BlockKey, int], int]]:
    data_by_group: dict[tuple[str, BlockKey], dict[int, float]] = {}
    run_ids: dict[tuple[str, BlockKey, int], int] = {}
    for block_key, block_dataset in block_datasets:
        scores = run_scores(block_dataset.metrics, metric_name, measure).select(["run_id", "value"])
        joined = scores.join(block_dataset.runs.select(["run_id", "seed", "condition_name"]), on="run_id", how="inner")
        for row in joined.iter_rows(named=True):
            condition, seed = str(row["condition_name"]), int(row["seed"])
            if (condition, block_key, seed) in run_ids:
                cell = f"condition {condition!r} in block {_block_label(block_key)!r}"
                raise duplicate_seed_error(cell, seed, (run_ids[(condition, block_key, seed)], row["run_id"]))
            run_ids[(condition, block_key, seed)] = int(row["run_id"])
            data_by_group.setdefault((condition, block_key), {})[seed] = float(row["value"])
    return data_by_group, run_ids


_ORDERED_VALUE_TYPES = frozenset({SweepValueType.INTEGER, SweepValueType.FLOAT})
_CHAMPION_METHOD = "hierarchical per-sample bootstrap, selection included"
_REUSED_RUNS_OMNIBUS_SKIPPED = (
    "shared referent runs appear in several blocks, so the blocks are not independent; "
    "each paired test stays within one block"
)
_CHAMPION_TESTS_SKIPPED = (
    "no permutation tests on tuned champions: the seeds that selected each champion would be reused to test it"
)


def _seeds_by_config(
    records: list[dict[str, Any]],
    conditions: list[str],
    condition_axis: str,
    tuned_axes: list[str],
    block_label: str,
    untuned: Sequence[str],
) -> dict[str, list[set[int]]]:
    """Seeds observed under each configuration of each condition; a tuned one needs two, an untuned one exactly one."""
    grouped: dict[str, dict[tuple[Any, ...], set[int]]] = {condition: {} for condition in conditions}
    for record in records:
        condition = record["hypers"][condition_axis]
        if condition in grouped:
            config = tuple(record["hypers"][axis] for axis in tuned_axes)
            grouped[condition].setdefault(config, set()).add(record["seed"])
    for condition, configs in grouped.items():
        if condition in untuned and len(configs) > 1:
            raise ValueError(
                f"condition {condition!r} is declared untuned but has {len(configs)} configurations of "
                f"{tuned_axes} in block {block_label!r}; an untuned referent has exactly one"
            )
        required = 1 if condition in untuned else 2
        if len(configs) < required:
            raise InsufficientDataError(
                f"condition {condition!r} in block {block_label!r} has {len(configs)} configuration(s) of "
                f"{tuned_axes}; a tuned-champion comparison needs at least two to select over, "
                "unless the condition is declared untuned",
                cells=(
                    InsufficientDataCell(
                        {"condition": condition, "block": block_label},
                        f"at least {required} configuration(s)",
                        f"{len(configs)} configuration(s)",
                    ),
                ),
            )
    return {condition: list(configs.values()) for condition, configs in grouped.items()}


def _champion_difference(
    block_dataset: ExperimentDataset,
    metric_name: str,
    seeds_by_config: dict[str, list[set[int]]],
    minuend: str,
    subtrahend: str,
    block_key: BlockKey,
    spec: BakeoffSpec,
    confidence: float,
    random_seed: int,
    higher_is_better: bool,
    measure: PerformanceMeasure,
    untuned: Sequence[str],
) -> PairedDifference:
    shared = set.intersection(*seeds_by_config[minuend], *seeds_by_config[subtrahend])
    skipped = PairedDifference(
        minuend=minuend,
        subtrahend=subtrahend,
        block=_block_label(block_key),
        estimate=None,
        ci=None,
        confidence=confidence,
        n_paired=len(shared),
        method=_champion_method((minuend, subtrahend), untuned),
        p_value=None,
        skipped_reason=f"only {len(shared)} seeds shared by every configuration, fewer than min_samples={spec.min_samples}",
    )
    if len(shared) < spec.min_samples:
        return skipped

    condition = block_dataset.dimensions.condition
    difference = _best_hypers_difference(
        block_dataset,
        metric_name,
        group_axis=condition,
        minuend=minuend,
        subtrahend=subtrahend,
        declared_axes={axis.name for axis in block_dataset.dimensions.sweep_axes},
        higher_is_better=higher_is_better,
        confidence_level=confidence,
        n_resamples=spec.n_resamples,
        min_samples=spec.min_samples,
        random_seed=random_seed,
        measure=measure,
        untuned=frozenset(untuned),
    )
    return replace(skipped, estimate=difference.mean_diff, ci=difference.estimate_ci, skipped_reason=None)


def _champion_method(pair: tuple[str, str], untuned: Sequence[str]) -> str:
    fixed = [condition for condition in pair if condition in untuned]
    if not fixed:
        return _CHAMPION_METHOD
    if len(fixed) == len(pair):
        return "untuned referents, no selection: plain paired percentile bootstrap of the mean over their seeds"
    tuned = next(condition for condition in pair if condition not in untuned)
    return (
        f"{_CHAMPION_METHOD} for {tuned}; untuned referent {fixed[0]}, no selection: "
        "plain paired percentile bootstrap over its seeds"
    )


def _paired_differences(
    scores: dict[tuple[str, BlockKey], dict[int, float]],
    pairs: list[tuple[str, str]],
    block_keys: list[BlockKey],
    spec: BakeoffSpec,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[PairedDifference, ...]:
    """Compare every planned pair within each block, never pooling runs across blocks."""
    planned = [
        _paired_difference(scores, minuend, subtrahend, block_key, spec, confidence, rng)
        for block_key in block_keys
        for minuend, subtrahend in pairs
    ]

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
        method=f"paired {spec.interval.replace('_', ' ')} bootstrap",
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
        interval=spec.interval,
        rng=rng,
    )
    return replace(
        skipped,
        estimate=interval.estimate,
        ci=(interval.ci_low, interval.ci_high),
        p_value=test.p_value,
        skipped_reason=None,
    )


def _omnibus_test(
    data_by_group: dict[tuple[str, BlockKey], dict[int, float]],
    conditions: list[str],
    block_keys: list[BlockKey],
    spec: BakeoffSpec,
    rng: np.random.Generator,
) -> StatisticalTestDetails:
    for block_key in block_keys:
        if not any(data_by_group.get((condition, block_key)) for condition in conditions):
            raise InsufficientDataError(
                f"bakeoff block {_block_label(block_key)!r} has no observed data",
                cells=(InsufficientDataCell({"block": _block_label(block_key)}, "at least one observed condition", "0"),),
            )

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
    # complete_case keeps only rows where every condition was observed; blockwise keeps
    # any row with at least two observed conditions, retaining usable incomplete blocks.
    required_observations = len(conditions) if spec.missingness == "complete_case" else 2
    valid_rows = observed_counts >= required_observations
    dropped_singleton_blocks = int(np.sum((observed_counts >= 1) & (observed_counts < required_observations)))
    matrix = matrix[valid_rows]
    if matrix.shape[0] < spec.min_samples:
        raise InsufficientDataError(
            f"need at least {spec.min_samples} blocks with at least two observed algorithms; "
            f"got {matrix.shape[0]}",
            cells=(
                InsufficientDataCell(
                    {}, f"at least {spec.min_samples} blocks with at least two observed algorithms", str(matrix.shape[0])
                ),
            ),
        )
    has_missing = bool(np.isnan(matrix).any())
    block_result = block_stratified_permutation_test(
        matrix,
        n_resamples=spec.n_resamples,
        min_blocks=spec.min_samples,
        rng=rng,
    )
    return StatisticalTestDetails(
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
    )


def _incomplete_cells(
    data_by_group: dict[tuple[str, BlockKey], dict[int, float]],
    conditions: list[str],
    block_keys: list[BlockKey],
    accounting: RunAccounting,
    shared: dict[str, tuple[str, ...]],
) -> tuple[IncompleteCell, ...]:
    incomplete: list[IncompleteCell] = []
    for block_key in block_keys:
        seeds = {condition: set(data_by_group.get((condition, block_key), {})) for condition in conditions}
        observed = set().union(*seeds.values())
        for condition in conditions:
            missing = sorted(observed - seeds[condition])
            if not missing:
                continue
            # A shared referent's accounting cell has no value on the axes it is shared across.
            own_axes = {axis: value for axis, value in block_key if condition not in shared.get(axis, ())}
            coordinates = {"condition_name": condition, **own_axes}
            planned = [
                item for item in accounting.cells if all(dict(item.cell).get(k) == v for k, v in coordinates.items())
            ]
            incomplete.append(
                IncompleteCell(
                    condition=condition,
                    block=_block_label(block_key),
                    missing_seeds=tuple(missing),
                    failed=sum(item.failed for item in planned),
                    pending=sum(item.pending for item in planned),
                    excluded=sum(sum(item.excluded.values()) for item in planned),
                )
            )
    return tuple(incomplete)


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
        ci_skipped_reason=None,
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
    if array.size >= 2:
        summary = replace(summary, sd=float(np.std(array, ddof=1)))
    if array.size < spec.min_samples:
        return replace(summary, ci_skipped_reason=f"only {array.size} runs, fewer than min_samples={spec.min_samples}")

    ci = bootstrap_scalar_ci(
        array,
        estimand=spec.estimand,
        confidence=spec.confidence_level,
        n_resamples=spec.n_resamples,
        min_samples=spec.min_samples,
        interval=spec.interval,
        rng=rng,
    )
    return replace(summary, ci=(ci.ci_low, ci.ci_high))


def _block_predicate(block_key: BlockKey, shared_across: Sequence[str]) -> pl.Expr:
    predicate = pl.lit(True)
    for column, value in block_key:
        matches = pl.col(column).eq_missing(value)
        if column in shared_across:
            matches = matches | pl.col(column).is_null()
        predicate = predicate & matches
    return predicate


def _block_keys(runs: pl.DataFrame, block_by: list[str], shared_across: Sequence[str]) -> list[BlockKey]:
    """Every observed block; a shared axis's blocks come only from the runs it applies to."""
    if not block_by:
        return [()]
    if shared_across:
        runs = runs.filter(pl.all_horizontal(*(pl.col(axis).is_not_null() for axis in shared_across)))
    rows = runs.select(block_by).unique().sort(block_by).iter_rows(named=True)
    return [tuple((column, row[column]) for column in block_by) for row in rows]


def _shared_conditions(
    dataset: ExperimentDataset, block_by: list[str], shared_across: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Each shared axis mapped to the conditions it does not apply to, whose runs join every one of its blocks."""
    conditional_axes = {axis.name for axis in dataset.dimensions.sweep_axes if axis.when is not None}
    invalid = sorted(axis for axis in shared_across if axis not in block_by or axis not in conditional_axes)
    if invalid:
        raise ValueError(f"shared_across must name block_by axes declared with a `when` rule, got {invalid}")

    shared: dict[str, tuple[str, ...]] = {}
    for axis in shared_across:
        by_condition = dataset.runs.group_by("condition_name").agg(
            pl.col(axis).is_null().all().alias("free"), pl.col(axis).is_null().any().alias("partly")
        )
        mixed = sorted(by_condition.filter(pl.col("partly") & ~pl.col("free"))["condition_name"].to_list())
        if mixed:
            raise ValueError(f"conditions {mixed} lack {axis!r} on only some runs; a shared referent must lack it on all")
        shared[axis] = tuple(sorted(by_condition.filter(pl.col("free"))["condition_name"].to_list()))
    return shared


def _require_every_run_placed(dataset: ExperimentDataset, blocks: list[tuple[BlockKey, ExperimentDataset]]) -> None:
    placed = {run_id for _key, block_dataset in blocks for run_id in block_dataset.runs["run_id"].to_list()}
    unplaced = dataset.runs.filter(~pl.col("run_id").is_in(sorted(placed)))
    if unplaced.is_empty():
        return
    counts = unplaced.group_by("condition_name").len().sort("condition_name").iter_rows()
    raise InsufficientDataError(
        f"{unplaced.height} run(s) match no block: a shared referent needs a block of the conditions that have the axis",
        cells=tuple(InsufficientDataCell({"condition": str(name)}, "a block to join", f"{count} unplaced run(s)") for name, count in counts),
    )


def _filter_dataset(dataset: ExperimentDataset, predicate: pl.Expr) -> ExperimentDataset:
    runs = dataset.runs.filter(predicate)
    metrics = dataset.metrics.join(runs.select("run_id"), on="run_id", how="inner")
    return replace(dataset, runs=runs, metrics=metrics)


def _filter_to_champions(
    dataset: ExperimentDataset, champions: dict[str, dict[str, Any]], untuned: Sequence[str]
) -> ExperimentDataset:
    """Restrict runs to each tuned condition's champion combination, keeping an untuned referent's only one."""
    condition_column = pl.col("condition_name")
    selected = pl.any_horizontal(
        condition_column.is_in(list(untuned)),
        *[
            condition_column.eq(condition) & pl.all_horizontal(*[pl.col(key).eq_missing(value) for key, value in config.items()])
            for condition, config in champions.items()
        ],
    )
    return _filter_dataset(dataset, selected)


def _planned_pairs(conditions: list[str], pairs: Sequence[tuple[str, str]] | None) -> list[tuple[str, str]]:
    if pairs is None:
        return [(minuend, subtrahend) for index, minuend in enumerate(conditions) for subtrahend in conditions[index + 1 :]]
    unknown = sorted({name for pair in pairs for name in pair} - set(conditions))
    if unknown:
        raise ValueError(f"pairs name conditions outside {conditions!r}: {unknown}")
    unordered = [frozenset(pair) for pair in pairs]
    if not unordered or any(len(pair) < 2 for pair in unordered) or len(set(unordered)) < len(unordered):
        raise ValueError(f"pairs must compare two distinct conditions, each pair once, got {list(pairs)!r}")
    return list(pairs)


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
    measure: PerformanceMeasure = FINAL_VALUE,
    pairs: Sequence[tuple[str, str]] | None = None,
    shared_across: Sequence[str] = (),
    untuned: Sequence[str] = (),
) -> BakeoffReport:
    """Champion selection, comparison, and statistical caveats, in one repeatable call.

    The headline is each block's paired, seed-matched difference for every planned pair of
    conditions (``pairs``; every pair when omitted), with percentile CIs Bonferroni-adjusted across the whole family; the omnibus
    and Holm-adjusted permutation p-values are secondary. With ``select_champion_over``, the
    differences come from the hierarchical per-sample bootstrap over each arm's grid, so they
    include selection uncertainty, and no tests are run on the selecting seeds.

    ``shared_across`` names ``block_by`` axes declared with a ``when`` rule. A condition the axis
    does not apply to has no block of its own on it; its runs join every block of that axis, so one
    axis-free referent is compared against each block's arms. Its runs are counted once in dataset
    accounting and listed once per block in ``runs``. Reused runs make blocks dependent, so the
    omnibus test is skipped; each paired test stays within one block.

    ``untuned`` declares conditions with a single configuration as fixed referents in a tuned-champion
    comparison: they have no selection stage, so their draws are the plain bootstrap over their seeds,
    paired by seed with the tuned arm's hierarchical draws. Any other condition with fewer than two
    configurations still fails.
    """
    if len(conditions) < 2:
        raise ValueError(f"bakeoff_report requires at least 2 conditions, got {len(conditions)} ({conditions!r})")
    planned_pairs = _planned_pairs(conditions, pairs)
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
    if untuned and select_champion_over is None:
        raise ValueError("untuned names fixed referents in a tuned-champion comparison; give select_champion_over")
    unknown_untuned = sorted(set(untuned) - set(conditions))
    if unknown_untuned:
        raise ValueError(f"untuned names conditions outside {conditions!r}: {unknown_untuned}")
    if set(conditions) <= set(untuned):
        raise ValueError("a tuned-champion comparison needs at least one tuned condition; every condition is untuned")

    dataset = _filter_dataset(dataset, pl.col("condition_name").is_in(conditions))

    # Only numeric axes have grid edges; string and boolean values carry no declared order.
    ordered_axes = [
        axis.name
        for axis in dataset.dimensions.sweep_axes
        if axis.name in (select_champion_over or ()) and axis.value_type in _ORDERED_VALUE_TYPES
    ]

    shared = _shared_conditions(dataset, block_by, shared_across)
    block_keys = _block_keys(dataset.runs, block_by, shared_across)
    blocks = [(block_key, _filter_dataset(dataset, _block_predicate(block_key, shared_across))) for block_key in block_keys]
    _require_every_run_placed(dataset, blocks)
    reuses_runs = sum(block_dataset.runs.height for _key, block_dataset in blocks) > dataset.runs.height

    if select_champion_over is not None and spec.estimand != "mean":
        raise ValueError("tuned-champion comparisons estimate a mean; spec.estimand must be 'mean'")
    if select_champion_over is not None and spec.interval != "percentile":
        raise ValueError("tuned-champion comparisons use the hierarchical percentile bootstrap; spec.interval must be 'percentile'")
    family_size = len(block_keys) * len(planned_pairs)
    comparison_confidence = bonferroni_confidence(spec.confidence_level, family_size)

    warnings: list[BakeoffWarning] = []
    champions: dict[str, dict[str, HyperparameterConfigReport]] = {}
    champion_differences: list[PairedDifference] = []
    compared_blocks: list[tuple[BlockKey, ExperimentDataset]] = []

    for block_key, block_dataset in blocks:
        cell = block_key

        if select_champion_over is None:
            compared_blocks.append((block_key, block_dataset))
            continue

        tuned_records = _load_dataset_hyperparameter_records(
            block_dataset, sorted({dataset.dimensions.condition, *select_champion_over}), metric_name, measure
        )
        seeds_by_config = _seeds_by_config(
            tuned_records,
            conditions,
            dataset.dimensions.condition,
            sorted(select_champion_over),
            _block_label(block_key),
            untuned,
        )
        for minuend, subtrahend in planned_pairs:
            champion_differences.append(
                _champion_difference(
                    block_dataset,
                    metric_name,
                    seeds_by_config,
                    minuend,
                    subtrahend,
                    block_key,
                    spec,
                    comparison_confidence,
                    spec.random_seed + len(champion_differences),
                    higher_is_better,
                    measure,
                    untuned,
                )
            )

        raw_champions = best_hypers(
            _filter_dataset(block_dataset, ~pl.col("condition_name").is_in(list(untuned))),
            metric_name,
            group_by=[dataset.dimensions.condition],
            higher_is_better=higher_is_better,
            confidence_level=spec.confidence_level,
            n_resamples=spec.n_resamples,
            min_samples=spec.min_samples,
            random_seed=spec.random_seed,
            measure=measure,
        )
        block_champions = {key[0]: report for key, report in raw_champions.items() if key[0] in conditions}
        champions[_block_label(block_key)] = block_champions

        for axis in ordered_axes:
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
                    vary=axis, values=sorted_values, best_performance=means, slice_performance=means
                )
                warning = boundary_warning(slice_report, condition=cond, cell=cell, higher_is_better=higher_is_better)
                if warning is not None:
                    warnings.append(warning)

        champion_dataset = _filter_to_champions(
            block_dataset, {condition: report.winning_config for condition, report in block_champions.items()}, untuned
        )
        compared_blocks.append((block_key, champion_dataset))

    data_by_group, run_ids = _load_bakeoff_data(compared_blocks, metric_name, measure)
    omnibus: StatisticalTestDetails | None = None
    omnibus_skipped_reason = None
    differences = tuple(champion_differences)
    if select_champion_over is None and spec.missingness == "error":
        incomplete = _incomplete_cells(data_by_group, conditions, block_keys, dataset.accounting, shared)
        if incomplete:
            raise IncompleteBlocksError(incomplete)
    if select_champion_over is None:
        test_rng = np.random.default_rng(spec.random_seed)
        if reuses_runs:
            omnibus_skipped_reason = _REUSED_RUNS_OMNIBUS_SKIPPED
        else:
            omnibus = _omnibus_test(data_by_group, conditions, block_keys, spec, test_rng)
        differences = _paired_differences(data_by_group, planned_pairs, block_keys, spec, comparison_confidence, test_rng)

    cells: dict[str, dict[str, CellSummary]] = {condition: {} for condition in conditions}
    cell_rng = np.random.default_rng(spec.random_seed)

    for block_key, block_dataset in compared_blocks:
        label = _block_label(block_key)
        for condition in conditions:
            seed_map = data_by_group.get((condition, block_key), {})
            values = list(seed_map.values())
            cell_run_ids = [run_ids[(condition, block_key, seed)] for seed in seed_map]
            cells[condition][label] = _summarize_cell(values, cell_run_ids, spec, cell_rng)
            coverage_issue = small_sample_coverage_warning(len(values), "cell", condition=condition, cell=block_key)
            if cells[condition][label].ci is not None and coverage_issue is not None:
                warnings.append(coverage_issue)

            if not values:
                continue
            variance_issue = variance_warning(values, condition=condition, cell=block_key)
            if variance_issue is not None:
                warnings.append(variance_issue)
            if variance_issue is None or variance_issue.code != BakeoffWarningCode.ZERO_VARIANCE:
                # Shapiro-Wilk is undefined (and noisy) on constant data; zero variance
                # is already reported above, so skip the redundant, degenerate check.
                normality_issue = non_normality_warning(values, condition=condition, cell=block_key)
                if normality_issue is not None:
                    warnings.append(normality_issue)
            if condition in champions.get(label, {}):
                seeds_issue = insufficient_seeds_warning(
                    champions[label][condition], len(values), condition=condition, cell=block_key, min_seeds=min_seeds
                )
                if seeds_issue is not None:
                    warnings.append(seeds_issue)

        flat = flatline_flags(block_dataset.metrics, metric_name)
        for row in flat.filter(pl.col("flatlined")).iter_rows(named=True):
            warnings.append(
                BakeoffWarning(
                    BakeoffWarningCode.FLATLINED_LEARNING_CURVE,
                    f"mean curve stops moving well before the run ends (tail range "
                    f"{row['tail_range']:.4g} of whole range {row['whole_range']:.4g}).",
                    row["condition_name"],
                    block_key,
                )
            )

    block_keys_by_label = {_block_label(block_key): block_key for block_key in block_keys}
    for item in differences:
        if item.ci is None:
            continue
        issue = small_sample_coverage_warning(
            item.n_paired,
            "paired difference",
            condition=f"{item.minuend} - {item.subtrahend}",
            cell=block_keys_by_label[item.block],
        )
        if issue is not None:
            warnings.append(issue)

    report = BakeoffReport(
        conditions=conditions,
        metric_name=metric_name,
        measure=measure,
        spec=spec,
        cells=cells,
        runs=tuple(
            RunScore(condition, _block_label(block_key), seed, run_ids[(condition, block_key, seed)], value)
            for (condition, block_key), seed_map in data_by_group.items()
            for seed, value in seed_map.items()
        ),
        champions=champions or None,
        blocks=[_block_label(block_key) for block_key in block_keys],
        omnibus=omnibus,
        tests_skipped_reason=None if select_champion_over is None else _CHAMPION_TESTS_SKIPPED,
        omnibus_skipped_reason=omnibus_skipped_reason,
        shared_across=shared,
        untuned=tuple(untuned),
        differences=differences,
        family_size=family_size,
        comparison_confidence=comparison_confidence,
        warnings=tuple(warnings),
    )

    return report
