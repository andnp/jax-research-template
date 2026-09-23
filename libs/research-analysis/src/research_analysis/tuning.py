from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research_analysis.duplicates import duplicate_seed_error


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
    if not np.isfinite(arr).all():
        raise ValueError("hyperparameter matrix has non-finite values; diverged runs need an explicit policy")


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
    and represent the same seed order -- matching the paired block-and-seed
    observations used by ``research_analysis.reporting.bakeoff_report`` to
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
    run_ids: dict[tuple[tuple[Any, ...], int], object] = {}
    for record in records:
        key = tuple(record["hypers"].get(k) for k in hyperparameter_keys)
        seed = record["seed"]
        if (key, seed) in run_ids:
            cell = f"configuration {dict(zip(hyperparameter_keys, key, strict=True))}"
            raise duplicate_seed_error(cell, seed, (run_ids[(key, seed)], record["run_id"]))
        run_ids[(key, seed)] = record["run_id"]
        groups.setdefault(key, {})[seed] = record["value"]

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

