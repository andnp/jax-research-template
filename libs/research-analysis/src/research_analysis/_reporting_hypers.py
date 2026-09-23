"""Private hyperparameter reporting implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import numpy as np

from research_analysis._reporting_console import (
    _present_best_hypers,
    _present_best_hypers_difference,
    _present_slice_hypers,
)
from research_analysis._reporting_models import (
    HyperparameterConfigReport,
    HyperparameterDifferenceReport,
    HyperparameterSliceReport,
)
from research_analysis._reporting_plots import _plot_hypers
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.non_finite import require_finite
from research_analysis.performance import FinalValue, PerformanceMeasure, run_scores
from research_analysis.tuning import _build_hyper_matrix, hierarchical_bootstrap_ci, hierarchical_bootstrap_difference_ci


def _load_dataset_hyperparameter_records(
    dataset: ExperimentDataset,
    hyperparameter_keys: Sequence[str],
    metric_name: str,
    measure: PerformanceMeasure,
) -> list[dict[str, Any]]:
    """Return per-run scores joined to the dataset's declared sweep axes and condition.

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

    scores = run_scores(dataset.metrics, metric_name, measure).select(["run_id", "value"])
    joined = scores.join(
        dataset.runs.select(["run_id", "seed", *run_columns.values()]), on=["run_id"], how="inner"
    )
    require_finite(joined, f"{metric_name!r} run scores")
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

    records = _load_dataset_hyperparameter_records(dataset, sorted(declared_axes | set(group_keys)), metric_name, FinalValue())

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

    hyper_matrix = _build_hyper_matrix(search_records, free_axes)
    matrix = hyper_matrix.values
    if len(hyper_matrix.seeds) < min_samples:
        raise ValueError(f"need at least {min_samples} seeds shared by every hyperparameter configuration")
    n_runs = matrix.shape[1]

    bootstrap = hierarchical_bootstrap_ci(
        matrix,
        n_runs,
        n_runs,
        confidence=confidence_level,
        n_resamples=n_resamples,
        rng=np.random.default_rng(random_seed),
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
        unpaired_run_ids=hyper_matrix.run_ids_outside(hyper_matrix.seeds),
    )

    if verbose:
        _present_best_hypers(report, experiment_slug, group_label)

    return report
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

    records = _load_dataset_hyperparameter_records(dataset, sorted(declared_axes | {group_axis}), metric_name, FinalValue())
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

    hyper_a = _build_hyper_matrix(records_a, free_axes)
    hyper_b = _build_hyper_matrix(records_b, free_axes)
    common_seeds = sorted(set(hyper_a.seeds) & set(hyper_b.seeds))
    if len(common_seeds) < min_samples:
        raise ValueError(f"need at least {min_samples} seeds shared by both compared groups")
    matrix_a = hyper_a.values[:, [hyper_a.seeds.index(seed) for seed in common_seeds]]
    matrix_b = hyper_b.values[:, [hyper_b.seeds.index(seed) for seed in common_seeds]]
    n_runs = matrix_a.shape[1]

    diff = hierarchical_bootstrap_difference_ci(
        matrix_a,
        matrix_b,
        n_runs,
        n_runs,
        confidence=confidence_level,
        n_resamples=n_resamples,
        rng=np.random.default_rng(random_seed),
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
        unpaired_run_ids=tuple(sorted(hyper_a.run_ids_outside(common_seeds) + hyper_b.run_ids_outside(common_seeds))),
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
    records = _load_dataset_hyperparameter_records(dataset, sorted(declared_axes | set(group_keys)), metric_name, FinalValue())

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


