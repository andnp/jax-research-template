"""Rank test for comparing two independent groups."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
from numpy.typing import NDArray

from research_analysis._kernels import mann_whitney_rank_sum
from research_analysis.non_finite import require_finite_array


@dataclass(frozen=True)
class MannWhitneyResult:
    """Result of a Mann–Whitney U-test.

    Attributes:
        u_statistic: The U statistic for group A.
        p_value: Two-sided p-value.
        rank_biserial_correlation: Effect size based on rank ordering.
        median_a: Sample median of group A.
        median_b: Sample median of group B.
        significant: Whether the difference is significant at the given alpha.
        alpha: Significance level used.
        n_a: Number of observations in group A.
        n_b: Number of observations in group B.
    """

    u_statistic: float
    p_value: float
    rank_biserial_correlation: float
    median_a: float
    median_b: float
    significant: bool
    alpha: float
    n_a: int
    n_b: int

def _as_float64_vector(values: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())


def _two_sided_p_value(u_statistic: float, n_a: int, n_b: int, tie_term: float) -> float:
    mean_u = n_a * n_b / 2.0
    n_total = n_a + n_b

    if n_total < 2:
        return 1.0

    tie_adjustment = tie_term / (n_total * (n_total - 1))
    variance = n_a * n_b * (n_total + 1.0 - tie_adjustment) / 12.0
    if variance <= 0.0:
        return 1.0

    distance = u_statistic - mean_u
    if distance > 0.0:
        z_score = (distance - 0.5) / sqrt(variance)
    elif distance < 0.0:
        z_score = (distance + 0.5) / sqrt(variance)
    else:
        z_score = 0.0

    return min(1.0, max(0.0, erfc(abs(z_score) / sqrt(2.0))))


def mann_whitney_u_test(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    alpha: float = 0.05,
) -> MannWhitneyResult:
    """Compare two independent groups with a Mann–Whitney U-test.


    Args:
        a: 1-D array of metric values for algorithm A.
        b: 1-D array of metric values for algorithm B.
        alpha: Significance level. Default 0.05.

    Returns:
        A :class:`MannWhitneyResult` with test statistics and significance.

    Raises:
        ValueError: If either group is empty or alpha is invalid.
    """
    a = _as_float64_vector(a)
    b = _as_float64_vector(b)

    if len(a) < 1:
        raise ValueError(f"Group A needs at least 1 observation, got {len(a)}")
    if len(b) < 1:
        raise ValueError(f"Group B needs at least 1 observation, got {len(b)}")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
    require_finite_array(a, "group A")
    require_finite_array(b, "group B")

    n_a = len(a)
    n_b = len(b)
    combined = np.concatenate((a, b))
    order = np.argsort(combined, kind="mergesort").astype(np.int64, copy=False)
    sorted_values = np.ascontiguousarray(combined[order])
    rank_sum_a, tie_term = mann_whitney_rank_sum(order, sorted_values, n_a)
    u_statistic = float(rank_sum_a - n_a * (n_a + 1) / 2.0)
    p_value = _two_sided_p_value(u_statistic, n_a, n_b, tie_term)
    rank_biserial_correlation = float(2.0 * u_statistic / (n_a * n_b) - 1.0)

    return MannWhitneyResult(
        u_statistic=u_statistic,
        p_value=p_value,
        rank_biserial_correlation=rank_biserial_correlation,
        median_a=float(np.median(a)),
        median_b=float(np.median(b)),
        significant=p_value < alpha,
        alpha=alpha,
        n_a=n_a,
        n_b=n_b,
    )
