from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HierarchicalBootstrapCI:
    """Result of hierarchical two-stage bootstrap hyperparameter evaluation."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    hyper_probs: NDArray[np.float64]
    selection_entropy: float


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
    if arr.ndim != 2:
        raise ValueError(f"data must be 2D array of shape (n_hypers, n_runs), got {arr.shape}")
    n_hypers, n_runs = arr.shape
    if n_hypers < 2:
        raise ValueError(f"need at least 2 hyperparameters to tune, got {n_hypers}")
    if n_runs < h_samples:
        raise ValueError(
            f"n_runs ({n_runs}) must be >= h_samples ({h_samples}): the selection budget is "
            "sampled without replacement, but evaluation resampling is with replacement, so "
            "e_samples may exceed n_runs"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

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


def hierarchical_bootstrap_ci_from_experiment(
    db_path: Path | str,
    experiment_slug: str,
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
    """Load an experiment's hyperparameter grid and evaluate it via hierarchical bootstrap.

    ``hyperparameter_keys`` names the axis (or axes, for a multi-dimensional
    grid) that define "which hyperparameter configuration" -- distinct
    combinations of their stored values become the rows tuned over. Matches
    the ``db_path``/``experiment_slug``/metric-name calling convention used by
    the rest of this package (e.g. ``analyze_hypers``) instead of requiring
    the caller to hand-assemble the raw matrix.
    """
    from research_analysis.reporting import _load_hyperparameter_records

    records = _load_hyperparameter_records(Path(db_path), experiment_slug, hyperparameter_keys, metric_name)
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

