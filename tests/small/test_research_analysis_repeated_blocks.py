"""Behavior tests for incomplete repeated-block inference."""

import numpy as np
import pytest
from research_analysis.repeated_blocks import (
    BlockPermutationResult,
    block_stratified_permutation_test,
)


def test_block_permutation_retains_incomplete_blocks() -> None:
    """Missing treatment cells remain in the blockwise permutation design."""
    scores = np.array(
        [[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [2.0, 3.0, np.nan], [1.0, 2.0, 4.0]]
    )

    result = block_stratified_permutation_test(
        scores, n_resamples=128, rng=np.random.default_rng(3)
    )

    assert isinstance(result, BlockPermutationResult)
    assert result.n_blocks == 4
    assert result.n_observations == 10
    assert 0.0 < result.p_value <= 1.0


def test_block_permutation_rejects_insufficient_observed_blocks() -> None:
    """A sparse matrix fails rather than returning a placeholder p-value."""
    scores = np.array([[1.0, np.nan, np.nan], [2.0, 3.0, np.nan]])

    with pytest.raises(ValueError, match="at least 3 blocks"):
        block_stratified_permutation_test(scores, n_resamples=16, rng=np.random.default_rng(0))


def test_block_permutation_rejects_infinite_scores() -> None:
    """Infinity is invalid even though NaN is the missing-cell marker."""
    scores = np.array([[1.0, 2.0], [np.inf, 3.0], [1.0, 4.0]])

    with pytest.raises(ValueError, match="infinity"):
        block_stratified_permutation_test(scores, n_resamples=16, rng=np.random.default_rng(0))
