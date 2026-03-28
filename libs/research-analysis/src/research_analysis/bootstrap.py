"""Bootstrap confidence intervals for learning curves.

Provides non-parametric bootstrap estimates of confidence intervals,
suitable for reporting RL learning curves across multiple seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from research_analysis._kernels import bootstrap_resample_means, mean_axis0, percentile_axis0

type BootstrapStatistic = NDArray[np.float64] | np.float64


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a bootstrap confidence interval estimation.

    Attributes:
        mean: Point estimate (mean across seeds).
        ci_low: Lower bound of the confidence interval.
        ci_high: Upper bound of the confidence interval.
        confidence: Confidence level (e.g. 0.95).
    """

    mean: BootstrapStatistic
    ci_low: BootstrapStatistic
    ci_high: BootstrapStatistic
    confidence: float


def _as_bootstrap_input(data: NDArray[np.floating]) -> tuple[NDArray[np.float64], bool]:
    array = np.asarray(data, dtype=np.float64)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[:, np.newaxis]

    return np.ascontiguousarray(array), squeeze


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
    working_data, squeeze = _as_bootstrap_input(data)

    n_seeds = working_data.shape[0]
    if n_seeds < 2:
        raise ValueError(f"Need at least 2 seeds for bootstrap, got {n_seeds}")
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    if rng is None:
        rng = np.random.default_rng()

    # Resample seed indices: (n_resamples, n_seeds)
    indices = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))
    # Bootstrap means: (n_resamples, n_steps)
    boot_means = bootstrap_resample_means(working_data, indices)
    sorted_boot_means = np.sort(boot_means, axis=0)

    alpha = 1.0 - confidence
    ci_low = percentile_axis0(sorted_boot_means, 100 * alpha / 2)
    ci_high = percentile_axis0(sorted_boot_means, 100 * (1 - alpha / 2))
    mean = mean_axis0(working_data)

    if squeeze:
        mean = np.float64(mean[0])
        ci_low = np.float64(ci_low[0])
        ci_high = np.float64(ci_high[0])

    return BootstrapCI(mean=mean, ci_low=ci_low, ci_high=ci_high, confidence=confidence)
