from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def get_hyper_distribution(
    data: NDArray[np.float64],
    h_samples: int,
    selection_trials: int,
    seed: int,
) -> tuple[NDArray[np.float64], float]:
    """Sample empirical probability distribution of argmax hyperparameter.

    Each trial's h_samples budget is drawn with replacement. h_samples is always
    called equal to n_runs in practice, and without-replacement sampling of a
    full pool of size n_runs is a fixed permutation with an invariant mean --
    every trial would recompute the exact same value and always pick the same
    winner, collapsing this distribution to a point mass regardless of how
    genuinely uncertain the true selection is.
    """
    np.random.seed(seed)
    n_hypers, n_runs = data.shape
    counts = np.zeros(n_hypers, dtype=np.float64)

    for _ in range(selection_trials):
        best_hyper = 0
        best_val = -1e30
        for h in range(n_hypers):
            total = 0.0
            for _ in range(h_samples):
                idx = np.random.randint(0, n_runs)
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

    A hyper is drawn independently for every sample, so each replicate's mean
    averages over the full selection-probability mixture rather than committing
    to one hyper. Tested under the e_samples == h_samples == n_runs precondition
    this function is always called under, this is better calibrated against the
    true best hyper's mean and produces narrower intervals than fixing one hyper
    per replicate -- a prior attempt to fix an apparent CI collapse by fixing the
    hyper per replicate was validated only outside that precondition (h_samples
    held fixed and small while e_samples was swept independently up to 5000),
    which is not how this function is ever actually called.
    """
    np.random.seed(seed)
    n_runs = data.shape[1]
    boot_means = np.empty(n_resamples, dtype=np.float64)

    cdf = np.cumsum(probs)
    cdf[-1] = 1.0

    for b in range(n_resamples):
        total = 0.0
        for _ in range(e_samples):
            h_idx = np.searchsorted(cdf, np.random.random())
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

    A hyper is drawn independently for every sample in each arm (see
    resample_two_stage), but both arms' e_samples draws reuse the same run
    index within a sample -- data_a and data_b must represent the same seed
    order in the same column positions -- to reduce variance from shared seeds
    the way research_analysis.reporting.compare_pairwise does.
    """
    np.random.seed(seed)
    n_runs = data_a.shape[1]
    diffs = np.empty(n_resamples, dtype=np.float64)

    cdf_a = np.cumsum(probs_a)
    cdf_a[-1] = 1.0
    cdf_b = np.cumsum(probs_b)
    cdf_b[-1] = 1.0

    for b in range(n_resamples):
        total_a = 0.0
        total_b = 0.0
        for _ in range(e_samples):
            run_idx = np.random.randint(0, n_runs)
            h_idx_a = np.searchsorted(cdf_a, np.random.random())
            h_idx_b = np.searchsorted(cdf_b, np.random.random())
            total_a += data_a[h_idx_a, run_idx]
            total_b += data_b[h_idx_b, run_idx]
        diffs[b] = total_a / e_samples - total_b / e_samples

    return diffs
