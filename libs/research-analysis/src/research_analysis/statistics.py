"""Statistical primitives for rigorous RL/ML evaluation.

Implements non-parametric tolerance intervals and representative run selection
based on the monorepo science guide standards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ToleranceInterval:
    """Result of a non-parametric tolerance interval estimation.

    Attributes:
        ci_low: Lower bound of the tolerance interval.
        ci_high: Upper bound of the tolerance interval.
        confidence: Confidence level (e.g. 0.95).
        coverage: Coverage level (e.g. 0.90).
    """

    ci_low: NDArray[np.float64] | np.float64
    ci_high: NDArray[np.float64] | np.float64
    confidence: float
    coverage: float


def tolerance_interval_confidence(num_samples: int, *, coverage: float, rank_span: int) -> float:
    """Calculate the achieved confidence for a given sample size, coverage, and rank span.

    Args:
        num_samples: Number of samples (seeds).
        coverage: Desired population coverage fraction in (0, 1).
        rank_span: The difference between the upper and lower order statistic ranks (j - i).

    Returns:
        The achieved confidence level.
    """
    return sum(
        math.comb(num_samples, failures)
        * (coverage**failures)
        * ((1.0 - coverage) ** (num_samples - failures))
        for failures in range(rank_span)
    )


def tolerance_interval_order_indices(
    num_samples: int, *, confidence: float = 0.95, coverage: float = 0.90
) -> tuple[int, int]:
    """Determine the rank-order indices for a non-parametric tolerance interval.

    Finds the smallest rank span j - i that achieves the target confidence.

    Args:
        num_samples: Number of samples (seeds).
        confidence: Desired confidence level in (0, 1). Default 0.95.
        coverage: Desired population coverage in (0, 1). Default 0.90.

    Returns:
        A tuple of (lower_index, upper_index) for indexing a sorted 0-indexed array.

    Raises:
        ValueError: If num_samples is too small, confidence/coverage are invalid,
            or if the target confidence cannot be achieved with the given sample size.
    """
    if num_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {num_samples}.")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if not 0 < coverage < 1:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}.")

    for rank_span in range(1, num_samples):
        achieved_confidence = tolerance_interval_confidence(
            num_samples, coverage=coverage, rank_span=rank_span
        )
        if achieved_confidence >= confidence:
            lower_index = (num_samples - 1 - rank_span) // 2
            upper_index = lower_index + rank_span
            return lower_index, upper_index

    raise ValueError(
        f"Insufficient samples ({num_samples}) to form a tolerance interval with "
        f"confidence={confidence} and coverage={coverage}."
    )


def pointwise_tolerance_interval(
    data: NDArray[np.floating],
    *,
    confidence: float = 0.95,
    coverage: float = 0.90,
) -> ToleranceInterval:
    """Compute non-parametric tolerance intervals for learning curves or scalar outcomes.

    Given data of shape (n_seeds, n_steps) or (n_seeds,), computes a pointwise
    tolerance interval by sorting the seeds and using order statistics.

    Args:
        data: Array of shape (n_seeds, n_steps) or (n_seeds,) containing metrics.
        confidence: Desired confidence level in (0, 1). Default 0.95.
        coverage: Desired population coverage fraction in (0, 1). Default 0.90.

    Returns:
        A ToleranceInterval containing the lower and upper bounds.

    Raises:
        ValueError: If data has fewer than 2 seeds or invalid confidence/coverage.
    """
    array = np.asarray(data, dtype=np.float64)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[:, np.newaxis]

    n_seeds = array.shape[0]
    lower_idx, upper_idx = tolerance_interval_order_indices(
        n_seeds, confidence=confidence, coverage=coverage
    )

    sorted_data = np.sort(array, axis=0)
    ci_low = sorted_data[lower_idx]
    ci_high = sorted_data[upper_idx]

    if squeeze:
        return ToleranceInterval(
            ci_low=np.float64(ci_low[0]),
            ci_high=np.float64(ci_high[0]),
            confidence=confidence,
            coverage=coverage,
        )

    return ToleranceInterval(
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        coverage=coverage,
    )


def select_median_run_index(data: NDArray[np.floating]) -> int:
    """Select the index of the seed that is closest to the median performance.

    For 2-D learning curves, compares the mean performance of each seed over time.
    Ties are broken by selecting the lower index.

    Args:
        data: Array of shape (n_seeds, n_steps) or (n_seeds,) containing metrics.

    Returns:
        The integer index of the representative median run.

    Raises:
        ValueError: If the input data is empty or has invalid shape.
    """
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        mean_scores = array
    elif array.ndim == 2:
        mean_scores = np.mean(array, axis=1)
    else:
        raise ValueError(f"Expected 1-D or 2-D array, got ndim={array.ndim}")

    if mean_scores.size == 0:
        raise ValueError("Input data must not be empty.")

    median_score = np.median(mean_scores)
    
    # Select the index of the run closest to the median score, breaking ties stably.
    min_diff = np.inf
    best_index = -1
    for idx, score in enumerate(mean_scores):
        diff = abs(score - median_score)
        if diff < min_diff:
            min_diff = diff
            best_index = idx
            
    return best_index
