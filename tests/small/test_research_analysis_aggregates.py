"""Behavior tests for fixed-reference benchmark aggregates."""

from collections.abc import Callable

import numpy as np
import pytest
from research_analysis.aggregates import (
    ReferenceRange,
    interquartile_mean,
    normalize_scores,
    optimality_gap,
    performance_profile,
    probability_of_improvement,
)


def test_normalization_uses_fixed_reference_direction() -> None:
    """Higher-is-better and lower-is-better metrics share the same orientation."""
    reference = ReferenceRange(low=0.0, high=10.0)
    lower_is_better = ReferenceRange(low=0.0, high=10.0, higher_is_better=False)

    assert np.allclose(normalize_scores(np.array([0.0, 5.0, 10.0]), reference), [0.0, 0.5, 1.0])
    assert np.allclose(normalize_scores(np.array([0.0, 5.0, 10.0]), lower_is_better), [1.0, 0.5, 0.0])


def test_profiles_improvement_and_gap_are_distribution_summaries() -> None:
    """Aggregate primitives preserve task-unit counts and ties."""
    scores = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.25]])

    profiles = performance_profile(scores, [0.5, 1.0], ["a", "b"])

    assert profiles["a"][0.5] == pytest.approx(2 / 3)
    assert profiles["a"][1.0] == pytest.approx(1 / 3)
    assert probability_of_improvement(scores[:, 0], scores[:, 1]) == pytest.approx(0.5)
    assert optimality_gap(scores[:, 0]) == pytest.approx(0.5)


def test_iqm_is_available_only_as_a_supplementary_statistic() -> None:
    """IQM computes a robust center while leaving dispersion to other summaries."""
    assert interquartile_mean(np.arange(1.0, 9.0)) == pytest.approx(4.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ReferenceRange(1.0, 1.0),
        lambda: normalize_scores(np.array([1.0, np.nan]), ReferenceRange(0.0, 1.0)),
        lambda: performance_profile(np.ones((2, 1)), [float("nan")], ["a"]),
        lambda: probability_of_improvement(np.ones(2), np.ones(3)),
        lambda: optimality_gap(np.array([])),
        lambda: interquartile_mean(np.array([1.0]), 0.5),
    ],
)
def test_aggregate_primitives_reject_invalid_inputs(factory: Callable[[], object]) -> None:
    """Invalid references, samples, thresholds, and shapes fail explicitly."""
    with pytest.raises(ValueError):
        factory()
