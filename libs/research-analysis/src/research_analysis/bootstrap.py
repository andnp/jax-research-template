"""Bootstrap confidence intervals for learning curves.

Provides non-parametric bootstrap estimates of confidence intervals,
suitable for reporting RL learning curves across multiple seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from research_analysis._kernels import bootstrap_resample_means, mean_axis0, percentile_axis0
from research_analysis.comparison import Estimand

type BootstrapStatistic = NDArray[np.float64] | np.float64
type ScalarBootstrapMethod = Literal["BCa", "degenerate-point"]


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


@dataclass(frozen=True)
class DifferenceBootstrapCI:
    """BCa interval for a scalar difference between two samples."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    pairing: Literal["paired", "independent"]
    estimand: Estimand
    method: ScalarBootstrapMethod = "BCa"


@dataclass(frozen=True)
class ScalarBootstrapCI:
    """BCa interval for one scalar sample."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    estimand: Estimand
    method: ScalarBootstrapMethod = "BCa"


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


def bootstrap_difference_ci(
    values_a: NDArray[np.floating],
    values_b: NDArray[np.floating],
    *,
    estimand: Estimand = "mean",
    pairing: Literal["paired", "independent"] = "independent",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    rng: np.random.Generator | None = None,
) -> DifferenceBootstrapCI:
    """Compute a BCa interval or label an exact constant-statistic point interval."""
    a = np.ascontiguousarray(np.asarray(values_a, dtype=np.float64).ravel())
    b = np.ascontiguousarray(np.asarray(values_b, dtype=np.float64).ravel())
    if estimand not in ("mean", "median"):
        raise ValueError(f"estimand must be 'mean' or 'median', got {estimand!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if min_samples < 3:
        raise ValueError(f"min_samples must be at least 3, got {min_samples}")
    if pairing == "paired" and len(a) != len(b):
        raise ValueError(f"paired samples must have equal lengths, got {len(a)} and {len(b)}")
    if len(a) < min_samples or len(b) < min_samples:
        raise ValueError(
            f"both samples need at least {min_samples} observations, got {len(a)} and {len(b)}"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bootstrap samples must contain only finite values")

    statistic = np.mean if estimand == "mean" else np.median

    def difference(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        return float(statistic(left) - statistic(right))

    estimate = difference(a, b)
    is_constant = (
        (pairing == "paired" and np.all((a - b) == (a - b)[0]))
        or (pairing == "independent" and np.all(a == a[0]) and np.all(b == b[0]))
    )
    if is_constant:
        return DifferenceBootstrapCI(
            estimate=estimate,
            ci_low=estimate,
            ci_high=estimate,
            confidence=confidence,
            n_resamples=n_resamples,
            pairing=pairing,
            estimand=estimand,
            method="degenerate-point",
        )

    if rng is None:
        rng = np.random.default_rng()
    result = stats.bootstrap(
        (a, b),
        difference,
        vectorized=False,
        paired=pairing == "paired",
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="BCa",
        rng=rng,
    )
    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("BCa bootstrap interval is undefined for these samples")
    return DifferenceBootstrapCI(
        estimate=estimate,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_resamples=n_resamples,
        pairing=pairing,
        estimand=estimand,
        method="BCa",
    )


def bootstrap_scalar_ci(
    values: NDArray[np.floating],
    *,
    estimand: Estimand = "mean",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    rng: np.random.Generator | None = None,
) -> ScalarBootstrapCI:
    """Compute a BCa interval or label an exact constant-statistic point interval."""
    data = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
    if estimand not in ("mean", "median"):
        raise ValueError(f"estimand must be 'mean' or 'median', got {estimand!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if min_samples < 3 or len(data) < min_samples:
        raise ValueError(f"need at least {min_samples} finite observations, got {len(data)}")
    if not np.isfinite(data).all():
        raise ValueError("bootstrap samples must contain only finite values")

    statistic = np.mean if estimand == "mean" else np.median

    def estimate(sample: NDArray[np.float64]) -> float:
        return float(statistic(sample))

    point_estimate = estimate(data)
    if np.all(data == data[0]):
        return ScalarBootstrapCI(
            estimate=point_estimate,
            ci_low=point_estimate,
            ci_high=point_estimate,
            confidence=confidence,
            n_resamples=n_resamples,
            estimand=estimand,
            method="degenerate-point",
        )

    if rng is None:
        rng = np.random.default_rng()
    result = stats.bootstrap(
        (data,),
        estimate,
        vectorized=False,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="BCa",
        rng=rng,
    )
    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("BCa bootstrap interval is undefined for these samples")
    return ScalarBootstrapCI(
        estimate=point_estimate,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_resamples=n_resamples,
        estimand=estimand,
        method="BCa",
    )
