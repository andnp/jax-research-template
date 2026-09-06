from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from research_analysis.dataset import ExperimentDataset


@dataclass(frozen=True)
class HierarchicalBootstrapCI:
    """Result of hierarchical two-stage bootstrap hyperparameter evaluation."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    hyper_probs: NDArray[np.float64]
    selection_entropy: float


def _validate_hyper_matrix(arr: NDArray[np.float64], confidence: float) -> None:
    if arr.ndim != 2:
        raise ValueError(f"data must be 2D array of shape (n_hypers, n_runs), got {arr.shape}")
    n_hypers, _n_runs = arr.shape
    if n_hypers < 2:
        raise ValueError(f"need at least 2 hyperparameters to tune, got {n_hypers}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def hierarchical_bootstrap_ci(
    data: NDArray[np.floating],
    h_samples: int,
    e_samples: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    selection_trials: int = 200,
    seed: int = 42,
) -> HierarchicalBootstrapCI:
    """Evaluate hyperparameter selection with hierarchical bootstrap resampling.

    Estimates the empirical probability distribution of the maximizing hyperparameter
    under budget ``h_samples``, and evaluates performance by sampling evaluation seeds
    from that induced distribution to prevent confidence interval collapse.
    """
    arr = np.asarray(data, dtype=np.float64)
    _validate_hyper_matrix(arr, confidence)

    from research_analysis._tuning_kernels import get_hyper_distribution, resample_two_stage

    probs, entropy = get_hyper_distribution(arr, h_samples, selection_trials, seed)
    boot_means = resample_two_stage(arr, probs, e_samples, n_resamples, seed + 1)

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2)))
    mean_val = float(np.mean(boot_means))

    return HierarchicalBootstrapCI(
        mean=mean_val,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        hyper_probs=probs,
        selection_entropy=entropy,
    )


@dataclass(frozen=True)
class HierarchicalBootstrapDifferenceCI:
    """Result of a paired hierarchical two-stage bootstrap difference evaluation (arm A minus arm B)."""

    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float
    hyper_probs_a: NDArray[np.float64]
    hyper_probs_b: NDArray[np.float64]
    selection_entropy_a: float
    selection_entropy_b: float


def hierarchical_bootstrap_difference_ci(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    h_samples: int,
    e_samples: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    selection_trials: int = 200,
    seed: int = 42,
) -> HierarchicalBootstrapDifferenceCI:
    """Evaluate the difference (arm A minus arm B) between two independently-tuned grids.

    Each arm selects its champion hyperparameter independently (its own
    stage-1 draw), then evaluation samples are paired by run index across
    arms, so ``data_a`` and ``data_b`` must have the same number of columns
    and represent the same seed order -- matching how
    ``research_analysis.reporting.compare_pairwise`` pairs seeded runs to
    reduce variance.
    """
    arr_a = np.asarray(data_a, dtype=np.float64)
    arr_b = np.asarray(data_b, dtype=np.float64)
    _validate_hyper_matrix(arr_a, confidence)
    _validate_hyper_matrix(arr_b, confidence)
    if arr_a.shape[1] != arr_b.shape[1]:
        raise ValueError(
            f"data_a and data_b must have the same number of paired runs, got {arr_a.shape[1]} and {arr_b.shape[1]}"
        )

    from research_analysis._tuning_kernels import get_hyper_distribution, resample_two_stage_difference

    probs_a, entropy_a = get_hyper_distribution(arr_a, h_samples, selection_trials, seed)
    probs_b, entropy_b = get_hyper_distribution(arr_b, h_samples, selection_trials, seed + 1)
    diffs = resample_two_stage_difference(arr_a, probs_a, arr_b, probs_b, e_samples, n_resamples, seed + 2)

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1.0 - alpha / 2)))
    mean_diff = float(np.mean(diffs))

    return HierarchicalBootstrapDifferenceCI(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        hyper_probs_a=probs_a,
        hyper_probs_b=probs_b,
        selection_entropy_a=entropy_a,
        selection_entropy_b=entropy_b,
    )


def _sortable_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    """Coerce a heterogeneous hyperparameter-value tuple into a deterministic sort key."""
    return tuple(str(value) for value in key)


def _build_hyper_matrix(
    records: list[dict[str, Any]],
    hyperparameter_keys: Sequence[str],
) -> tuple[NDArray[np.float64], list[tuple[Any, ...]], list[int]]:
    """Pivot loaded run records into a (n_hypers, n_runs) matrix, paired and ordered by seed.

    Runs are intersected down to the seeds shared by every hyperparameter
    combination so the matrix stays rectangular and each column is the same
    seed across rows -- required both for a valid hierarchical bootstrap and
    for pairing two arms' matrices in a difference evaluation.
    """
    groups: dict[tuple[Any, ...], dict[int, float]] = {}
    for record in records:
        key = tuple(record["hypers"].get(k) for k in hyperparameter_keys)
        groups.setdefault(key, {})[record["seed"]] = record["value"]

    if len(groups) < 2:
        raise ValueError(
            f"need at least 2 distinct combinations of {list(hyperparameter_keys)!r} to tune, got {len(groups)}"
        )

    keys_sorted = sorted(groups, key=_sortable_key)
    common_seeds = set.intersection(*(set(g) for g in groups.values()))
    if not common_seeds:
        raise ValueError(
            f"no seed has a completed run under every combination of {list(hyperparameter_keys)!r}"
        )
    seeds_sorted = sorted(common_seeds)

    matrix = np.array([[groups[key][s] for s in seeds_sorted] for key in keys_sorted], dtype=np.float64)
    return matrix, keys_sorted, seeds_sorted


def _all_keys(primary: Sequence[str], condition: Mapping[str, Any] | None) -> list[str]:
    """Union of the tuned keys and any fixed-condition keys, tuned keys first, no duplicates."""
    keys: list[str] = list(primary)
    for key in condition or {}:
        if key not in keys:
            keys.append(key)
    return keys


def _filter_condition(
    records: list[dict[str, Any]], condition: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Restrict records to those whose hypers match every key/value in condition."""
    if not condition:
        return records
    return [record for record in records if all(record["hypers"].get(k) == v for k, v in condition.items())]


def _dataset_hyperparameter_records(
    dataset: ExperimentDataset,
    hyperparameter_keys: Sequence[str],
    metric_name: str,
) -> list[dict[str, Any]]:
    run_columns = {
        key: "condition_name" if key == dataset.dimensions.condition else key for key in hyperparameter_keys
    }
    missing_keys = [key for key, column in run_columns.items() if column not in dataset.runs.columns]
    if missing_keys:
        raise ValueError(f"No declared dataset dimensions found for {missing_keys!r}")

    final_metrics = (
        dataset.metrics.filter(pl.col("metric") == metric_name)
        .sort(["run_id", "execution_id", "step"])
        .group_by(["run_id", "execution_id"])
        .agg(pl.col("value").last().alias("value"))
    )
    joined = final_metrics.join(
        dataset.runs.select(["run_id", "seed", *run_columns.values()]),
        on="run_id",
        how="inner",
    )
    records = [
        {
            "run_id": row["run_id"],
            "seed": row["seed"],
            "value": row["value"],
            "hypers": {key: row[column] for key, column in run_columns.items()},
        }
        for row in joined.iter_rows(named=True)
    ]
    if not records:
        raise ValueError(f"No completed runs found containing hyperparameters {list(hyperparameter_keys)!r}")
    return records


def hierarchical_bootstrap_ci_from_experiment(
    dataset: ExperimentDataset,
    hyperparameter_keys: Sequence[str],
    metric_name: str,
    h_samples: int,
    e_samples: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    selection_trials: int = 200,
    seed: int = 42,
) -> HierarchicalBootstrapCI:
    """Evaluate a declared dataset hyperparameter grid via hierarchical bootstrap."""
    records = _dataset_hyperparameter_records(dataset, hyperparameter_keys, metric_name)
    matrix, _keys, _seeds = _build_hyper_matrix(records, hyperparameter_keys)
    return hierarchical_bootstrap_ci(
        matrix,
        h_samples,
        e_samples,
        confidence=confidence,
        n_resamples=n_resamples,
        selection_trials=selection_trials,
        seed=seed,
    )


def hierarchical_bootstrap_difference_ci_from_experiment(
    dataset: ExperimentDataset,
    hyperparameter_keys_a: Sequence[str],
    hyperparameter_keys_b: Sequence[str],
    metric_name: str,
    h_samples: int,
    e_samples: int,
    *,
    condition_a: Mapping[str, Any] | None = None,
    condition_b: Mapping[str, Any] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    selection_trials: int = 200,
    seed: int = 42,
) -> HierarchicalBootstrapDifferenceCI:
    """Evaluate two independently tuned dataset arms via paired bootstrap."""
    all_keys_a = _all_keys(hyperparameter_keys_a, condition_a)
    all_keys_b = _all_keys(hyperparameter_keys_b, condition_b)

    records_a = _filter_condition(
        _dataset_hyperparameter_records(dataset, all_keys_a, metric_name), condition_a
    )
    records_b = _filter_condition(
        _dataset_hyperparameter_records(dataset, all_keys_b, metric_name), condition_b
    )

    matrix_a, _keys_a, seeds_a = _build_hyper_matrix(records_a, hyperparameter_keys_a)
    matrix_b, _keys_b, seeds_b = _build_hyper_matrix(records_b, hyperparameter_keys_b)

    common_seeds = sorted(set(seeds_a) & set(seeds_b))
    if not common_seeds:
        raise ValueError("no seed has a completed run under every combination of both arms")
    matrix_a = matrix_a[:, [seeds_a.index(s) for s in common_seeds]]
    matrix_b = matrix_b[:, [seeds_b.index(s) for s in common_seeds]]

    return hierarchical_bootstrap_difference_ci(
        matrix_a,
        matrix_b,
        h_samples,
        e_samples,
        confidence=confidence,
        n_resamples=n_resamples,
        selection_trials=selection_trials,
        seed=seed,
    )
