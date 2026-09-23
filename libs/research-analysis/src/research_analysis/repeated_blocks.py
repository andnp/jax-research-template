"""Blockwise permutation inference for incomplete repeated measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


@dataclass(frozen=True)
class BlockPermutationResult:
    """Result of a block-stratified omnibus permutation test."""

    statistic: float
    p_value: float
    n_blocks: int
    n_observations: int
    n_resamples: int


def _rank_score(matrix: NDArray[np.float64]) -> float:
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    counts = np.zeros(matrix.shape[1], dtype=np.float64)
    for row in matrix:
        observed = np.flatnonzero(np.isfinite(row))
        ranks = rankdata(row[observed], method="average")
        centered = ranks - (len(observed) + 1.0) / 2.0
        scores[observed] += centered
        counts[observed] += 1.0
    return float(np.sum(np.divide(scores**2, counts, out=np.zeros_like(scores), where=counts > 0)))


def block_stratified_permutation_test(
    scores: NDArray[np.floating],
    *,
    n_resamples: int = 10_000,
    min_blocks: int = 3,
    rng: np.random.Generator,
) -> BlockPermutationResult:
    """Test repeated blocks while retaining their distinct missingness patterns.

    NaN marks a missing cell, so it cannot also stand for a diverged run: reject or substitute
    non-finite run values before building ``scores`` (``bakeoff_report`` rejects them).
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("scores must be a 2-D matrix with at least two treatments")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if min_blocks < 3:
        raise ValueError(f"min_blocks must be at least 3, got {min_blocks}")
    if np.isinf(matrix).any():
        raise ValueError("scores may use NaN for missing cells but not infinity")
    observed_counts = np.isfinite(matrix).sum(axis=1)
    valid_rows = observed_counts >= 2
    if int(valid_rows.sum()) < min_blocks:
        raise ValueError(
            f"need at least {min_blocks} blocks with two observations, got {int(valid_rows.sum())}"
        )
    matrix = matrix[valid_rows]
    observed_statistic = _rank_score(matrix)
    exceedances = 0
    for _ in range(n_resamples):
        shuffled = matrix.copy()
        for row in shuffled:
            observed_indices = np.flatnonzero(np.isfinite(row))
            row[observed_indices] = row[observed_indices][rng.permutation(len(observed_indices))]
        exceedances += _rank_score(shuffled) >= observed_statistic
    return BlockPermutationResult(
        statistic=observed_statistic,
        p_value=float((exceedances + 1) / (n_resamples + 1)),
        n_blocks=matrix.shape[0],
        n_observations=int(np.isfinite(matrix).sum()),
        n_resamples=n_resamples,
    )
