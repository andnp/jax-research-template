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
    """Sample empirical probability distribution of argmax hyperparameter."""
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
    """Resample evaluation seeds according to the hyperparameter probability distribution."""
    np.random.seed(seed)
    n_runs = data.shape[1]
    boot_means = np.empty(n_resamples, dtype=np.float64)

    cdf = np.cumsum(probs)
    cdf[-1] = 1.0

    for b in range(n_resamples):
        total = 0.0
        for _ in range(e_samples):
            # Sample hyper from distribution
            r = np.random.random()
            h_idx = np.searchsorted(cdf, r)
            # Sample run for that hyper with replacement
            run_idx = np.random.randint(0, n_runs)
            total += data[h_idx, run_idx]
        boot_means[b] = total / e_samples

    return boot_means
