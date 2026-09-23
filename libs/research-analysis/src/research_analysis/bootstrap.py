"""Bootstrap confidence intervals for learning curves.

Provides non-parametric bootstrap estimates of confidence intervals,
suitable for reporting RL learning curves across multiple seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from research_analysis._kernels import bootstrap_resample_means
from research_analysis.aggregates import interquartile_mean
from research_analysis.comparison import Estimand
from research_analysis.non_finite import require_finite_array

type BootstrapStatistic = NDArray[np.float64] | np.float64
type PercentileBootstrapStatistic = Literal["mean", "interquartile_mean"]


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
    """Percentile-bootstrap interval for a scalar difference between two samples."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    pairing: Literal["paired", "independent"]
    estimand: Estimand


@dataclass(frozen=True)
class ScalarBootstrapCI:
    """Percentile-bootstrap interval for one scalar sample."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    estimand: Estimand


@dataclass(frozen=True)
class PercentileBootstrapCI:
    """Percentile interval for one scalar sample."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    statistic: PercentileBootstrapStatistic


def resolve_block_size(n_points: int, block_size: int | None = None) -> int:
    """Return the configured or cube-root moving-block length."""
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    if block_size is not None and (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer when provided")
    return max(1, round(n_points ** (1.0 / 3.0))) if block_size is None else block_size


def circular_moving_block_resample(
    residuals: NDArray[np.floating],
    *,
    block_size: int | None = None,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Resample centered residuals in circular contiguous blocks."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("residuals must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("residuals must contain only finite values")
    chosen_block_size = resolve_block_size(int(values.size), block_size)
    centered = values - np.mean(values)
    n_blocks = math.ceil(values.size / chosen_block_size)
    starts = rng.integers(0, values.size, size=n_blocks)
    offsets = np.arange(chosen_block_size, dtype=np.int64)
    indices = ((starts[:, None] + offsets) % values.size).reshape(-1)[: values.size]
    return np.ascontiguousarray(centered[indices])


def bootstrap_percentile_ci(
    values: NDArray[np.floating],
    *,
    statistic: PercentileBootstrapStatistic = "mean",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> PercentileBootstrapCI:
    """Compute a percentile interval for a scalar sample statistic."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size < 2 or not np.isfinite(observations).all():
        raise ValueError("bootstrap values must contain at least two finite observations")
    if statistic not in ("mean", "interquartile_mean"):
        raise ValueError(f"unsupported percentile bootstrap statistic: {statistic!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be at least two")

    if rng is None:
        rng = np.random.default_rng()
    indices = rng.integers(0, observations.size, size=(n_resamples, observations.size))
    samples = observations[indices]
    if statistic == "mean":
        estimates = np.mean(samples, axis=1)
        estimate = float(np.mean(observations))
    else:
        estimates = np.asarray([interquartile_mean(sample) for sample in samples], dtype=np.float64)
        estimate = interquartile_mean(observations)

    alpha = (1.0 - confidence) / 2.0
    return PercentileBootstrapCI(
        estimate=estimate,
        ci_low=float(np.quantile(estimates, alpha)),
        ci_high=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        n_resamples=n_resamples,
        statistic=statistic,
    )


def _percentile_interval(
    replicates: NDArray[np.float64], confidence: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the percentile-bootstrap bounds of ``replicates`` along the resample axis."""
    alpha = (1.0 - confidence) / 2.0
    return np.quantile(replicates, alpha, axis=0), np.quantile(replicates, 1.0 - alpha, axis=0)


def _sample_statistic(samples: NDArray[np.float64], estimand: Estimand) -> NDArray[np.float64]:
    """Apply the estimand across each row of ``samples``."""
    return np.mean(samples, axis=1) if estimand == "mean" else np.median(samples, axis=1)


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
    require_finite_array(working_data, "bootstrap_ci data")

    if rng is None:
        rng = np.random.default_rng()

    # Resample seed indices: (n_resamples, n_seeds)
    indices = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))
    # Bootstrap means: (n_resamples, n_steps)
    boot_means = bootstrap_resample_means(working_data, indices)
    ci_low, ci_high = _percentile_interval(boot_means, confidence)
    mean = np.mean(working_data, axis=0)

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
    """Compute a percentile-bootstrap interval for a difference, resampling runs jointly when paired."""
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

    if rng is None:
        rng = np.random.default_rng()
    indices_a = rng.integers(0, len(a), size=(n_resamples, len(a)))
    indices_b = indices_a if pairing == "paired" else rng.integers(0, len(b), size=(n_resamples, len(b)))
    replicates = _sample_statistic(a[indices_a], estimand) - _sample_statistic(b[indices_b], estimand)
    ci_low, ci_high = _percentile_interval(replicates, confidence)
    estimate = _sample_statistic(a[np.newaxis], estimand)[0] - _sample_statistic(b[np.newaxis], estimand)[0]
    return DifferenceBootstrapCI(
        estimate=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_resamples=n_resamples,
        pairing=pairing,
        estimand=estimand,
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
    """Compute a percentile-bootstrap interval for one scalar sample statistic."""
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

    if rng is None:
        rng = np.random.default_rng()
    indices = rng.integers(0, len(data), size=(n_resamples, len(data)))
    ci_low, ci_high = _percentile_interval(_sample_statistic(data[indices], estimand), confidence)
    return ScalarBootstrapCI(
        estimate=float(_sample_statistic(data[np.newaxis], estimand)[0]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_resamples=n_resamples,
        estimand=estimand,
    )
