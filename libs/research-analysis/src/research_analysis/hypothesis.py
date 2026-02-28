"""Hypothesis testing for algorithm comparison.

Provides Welch's t-test for comparing mean performance of two algorithms,
suitable for RL experiments with unequal variance across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class WelchResult:
    """Result of a Welch's t-test.

    Attributes:
        t_statistic: The t-test statistic.
        p_value: Two-sided p-value.
        df: Degrees of freedom (Welch–Satterthwaite approximation).
        mean_a: Sample mean of group A.
        mean_b: Sample mean of group B.
        significant: Whether the difference is significant at the given alpha.
        alpha: Significance level used.
    """

    t_statistic: float
    p_value: float
    df: float
    mean_a: float
    mean_b: float
    significant: bool
    alpha: float


def welch_ttest(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    *,
    alpha: float = 0.05,
) -> WelchResult:
    """Welch's t-test for comparing two independent groups.

    This is the correct test for comparing RL algorithms across seeds
    because it does not assume equal variance between groups.

    Args:
        a: 1-D array of metric values for algorithm A (e.g. final returns
            per seed).
        b: 1-D array of metric values for algorithm B.
        alpha: Significance level. Default 0.05.

    Returns:
        A :class:`WelchResult` with test statistics and significance.

    Raises:
        ValueError: If either group has fewer than 2 observations.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if len(a) < 2:
        raise ValueError(f"Group A needs at least 2 observations, got {len(a)}")
    if len(b) < 2:
        raise ValueError(f"Group B needs at least 2 observations, got {len(b)}")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    result = stats.ttest_ind(a, b, equal_var=False)

    # Welch–Satterthwaite degrees of freedom
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = numerator / denominator if denominator > 0 else 0.0

    return WelchResult(
        t_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=float(df),
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        significant=float(result.pvalue) < alpha,
        alpha=alpha,
    )
