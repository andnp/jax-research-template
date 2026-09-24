"""Permutation tests for independent and paired outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from research_analysis.comparison import Estimand, Pairing


@dataclass(frozen=True)
class PermutationResult:
    """Result of a declared permutation comparison."""

    statistic: float
    p_value: float
    estimand: Estimand
    pairing: Pairing
    alpha: float
    n_resamples: int

    @property
    def is_significant(self) -> bool:
        """Return whether the p-value is below the declared alpha."""
        return self.p_value < self.alpha


def run_permutation_test(
    values_a: NDArray[np.floating],
    values_b: NDArray[np.floating],
    *,
    estimand: Estimand = "mean",
    pairing: Pairing = "independent",
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    min_samples: int = 3,
    rng: np.random.Generator,
) -> PermutationResult:
    """Compare two finite samples using a declared permutation design.

    Paired, the statistic is the estimand of the per-seed differences and each pair's labels
    are swapped, a sign-flip test; independent, it is the difference of each group's estimand.
    """
    a = np.ascontiguousarray(np.asarray(values_a, dtype=np.float64).ravel())
    b = np.ascontiguousarray(np.asarray(values_b, dtype=np.float64).ravel())
    if estimand not in ("mean", "median"):
        raise ValueError(f"estimand must be 'mean' or 'median', got {estimand!r}")
    if pairing not in ("paired", "independent"):
        raise ValueError(f"permutation pairing must be paired or independent, got {pairing!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if min_samples < 3:
        raise ValueError(f"min_samples must be at least 3, got {min_samples}")
    if pairing == "paired" and len(a) != len(b):
        raise ValueError(f"paired groups must have equal lengths, got {len(a)} and {len(b)}")
    if len(a) < min_samples or len(b) < min_samples:
        raise ValueError(
            f"both groups need at least {min_samples} observations, got {len(a)} and {len(b)}"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("permutation samples must contain only finite values")

    statistic = np.mean if estimand == "mean" else np.median

    def paired_difference(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        return float(statistic(left - right))

    def independent_difference(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        return float(statistic(left) - statistic(right))

    paired = pairing == "paired"
    result = stats.permutation_test(
        (a, b),
        paired_difference if paired else independent_difference,
        permutation_type="samples" if paired else "independent",
        alternative="two-sided",
        n_resamples=n_resamples,
        rng=rng,
    )
    return PermutationResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        estimand=estimand,
        pairing=pairing,
        alpha=alpha,
        n_resamples=n_resamples,
    )
