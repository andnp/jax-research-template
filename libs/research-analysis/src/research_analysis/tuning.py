from __future__ import annotations

from dataclasses import dataclass

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

