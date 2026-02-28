"""Bootstrap confidence intervals for learning curves.

Provides non-parametric bootstrap estimates of confidence intervals,
suitable for reporting RL learning curves across multiple seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a bootstrap confidence interval estimation.

    Attributes:
        mean: Point estimate (mean across seeds).
        ci_low: Lower bound of the confidence interval.
        ci_high: Upper bound of the confidence interval.
        confidence: Confidence level (e.g. 0.95).
    """

    mean: NDArray[np.floating]
    ci_low: NDArray[np.floating]
    ci_high: NDArray[np.floating]
    confidence: float


def bootstrap_ci(
    data: NDArray[np.floating],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute bootstrap confidence intervals for learning curves.

    Given data of shape ``(n_seeds, n_steps)``, computes a pointwise
    confidence interval at each step by resampling seeds with replacement.

    Args:
        data: Array of shape ``(n_seeds, n_steps)`` or ``(n_seeds,)``
            containing metric values (e.g. episodic returns).
        confidence: Confidence level in ``(0, 1)``. Default 0.95.
        n_resamples: Number of bootstrap resamples.
        rng: NumPy random generator. Defaults to unseeded.

    Returns:
        A :class:`BootstrapCI` with arrays of shape ``(n_steps,)`` or
        scalar for 1-D input.

    Raises:
        ValueError: If data has fewer than 2 seeds or invalid confidence.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n_seeds = data.shape[0]
    if n_seeds < 2:
        raise ValueError(f"Need at least 2 seeds for bootstrap, got {n_seeds}")
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    if rng is None:
        rng = np.random.default_rng()

    # Resample seed indices: (n_resamples, n_seeds)
    indices = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))
    # Bootstrap means: (n_resamples, n_steps)
    boot_means = data[indices].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    mean = data.mean(axis=0)

    if squeeze:
        mean = mean.squeeze()
        ci_low = ci_low.squeeze()
        ci_high = ci_high.squeeze()

    return BootstrapCI(mean=mean, ci_low=ci_low, ci_high=ci_high, confidence=confidence)
