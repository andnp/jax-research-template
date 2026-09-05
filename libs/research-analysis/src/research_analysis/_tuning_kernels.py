from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def _sample_without_replacement(n_runs: int, k: int) -> NDArray[np.int64]:
    """Draw k distinct indices from range(n_runs) via partial Fisher-Yates."""
    idxs = np.arange(n_runs)
    for i in range(k):
        j = i + np.random.randint(0, n_runs - i)
        idxs[i], idxs[j] = idxs[j], idxs[i]
    return idxs[:k]


@numba.njit(cache=True)
def get_hyper_distribution(
    data: NDArray[np.float64],
    h_samples: int,
    selection_trials: int,
    seed: int,
) -> tuple[NDArray[np.float64], float]:
    """Sample empirical probability distribution of argmax hyperparameter.

    Each trial's h_samples budget is drawn without replacement, matching a
    real "I only ran h_samples distinct seeds" selection procedure rather
    than allowing the same seed to be counted twice within one trial.
    """
    np.random.seed(seed)
    n_hypers, n_runs = data.shape
    counts = np.zeros(n_hypers, dtype=np.float64)

    for _ in range(selection_trials):
        best_hyper = 0
        best_val = -1e30
        for h in range(n_hypers):
            total = 0.0
            trial_idxs = _sample_without_replacement(n_runs, h_samples)
            for idx in trial_idxs:
                total += data[h, idx]
            mean_val = total / h_samples
            if mean_val > best_val:
                best_val = mean_val
                best_hyper = h
        counts[best_hyper] += 1.0

    probs = counts / selection_trials
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * np.log(p)

    return probs, entropy


@numba.njit(cache=True)
def resample_two_stage(
    data: NDArray[np.float64],
    probs: NDArray[np.float64],
    e_samples: int,
    n_resamples: int,
    seed: int,
) -> NDArray[np.float64]:
    """Resample evaluation seeds according to the hyperparameter probability distribution.

    One hyper is drawn per replicate, not per sample: a real selection process
    picks one hyper and then evaluates it, so fixing the draw for the whole
    replicate preserves between-hyper selection variance across replicates.
    Redrawing per sample would average it away as e_samples grows, collapsing
    the reported interval regardless of how uncertain the selection actually is.
    """
    np.random.seed(seed)
    n_runs = data.shape[1]
    boot_means = np.empty(n_resamples, dtype=np.float64)

    cdf = np.cumsum(probs)
    cdf[-1] = 1.0

    for b in range(n_resamples):
        r = np.random.random()
        h_idx = np.searchsorted(cdf, r)
        total = 0.0
        for _ in range(e_samples):
            run_idx = np.random.randint(0, n_runs)
            total += data[h_idx, run_idx]
        boot_means[b] = total / e_samples

    return boot_means


@numba.njit(cache=True)
def resample_two_stage_difference(
    data_a: NDArray[np.float64],
    probs_a: NDArray[np.float64],
    data_b: NDArray[np.float64],
    probs_b: NDArray[np.float64],
    e_samples: int,
    n_resamples: int,
    seed: int,
) -> NDArray[np.float64]:
    """Paired hierarchical bootstrap of the difference mean_a - mean_b.

    Each arm draws its own hyper once per replicate (its own independent
    selection process), but both arms' e_samples draws reuse the same run
    index within a replicate -- data_a and data_b must represent the same
    seed order in the same column positions -- to reduce variance from shared
    seeds the way research_analysis.reporting.compare_pairwise does.
    """
    np.random.seed(seed)
    n_runs = data_a.shape[1]
    diffs = np.empty(n_resamples, dtype=np.float64)

    cdf_a = np.cumsum(probs_a)
    cdf_a[-1] = 1.0
    cdf_b = np.cumsum(probs_b)
    cdf_b[-1] = 1.0

    for b in range(n_resamples):
        h_idx_a = np.searchsorted(cdf_a, np.random.random())
        h_idx_b = np.searchsorted(cdf_b, np.random.random())
        total_a = 0.0
        total_b = 0.0
        for _ in range(e_samples):
            run_idx = np.random.randint(0, n_runs)
            total_a += data_a[h_idx_a, run_idx]
            total_b += data_b[h_idx_b, run_idx]
        diffs[b] = total_a / e_samples - total_b / e_samples

    return diffs
