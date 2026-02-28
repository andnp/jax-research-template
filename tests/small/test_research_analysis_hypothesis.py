"""Small tests for research_analysis.hypothesis — Welch's t-test."""

import numpy as np
import pytest
from research_analysis.hypothesis import WelchResult, welch_ttest


class TestWelchTtest:
    def test_identical_groups_not_significant(self):
        rng = np.random.default_rng(42)
        a = rng.normal(loc=0, scale=1, size=50)
        b = rng.normal(loc=0, scale=1, size=50)
        result = welch_ttest(a, b)
        assert isinstance(result, WelchResult)
        # Same distribution — should usually not be significant
        assert result.p_value > 0.01

    def test_different_groups_significant(self):
        rng = np.random.default_rng(42)
        a = rng.normal(loc=0, scale=0.1, size=100)
        b = rng.normal(loc=5, scale=0.1, size=100)
        result = welch_ttest(a, b)
        assert result.significant
        assert result.p_value < 0.001

    def test_means_correct(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = welch_ttest(a, b)
        assert abs(result.mean_a - 2.0) < 1e-10
        assert abs(result.mean_b - 5.0) < 1e-10

    def test_custom_alpha(self):
        rng = np.random.default_rng(42)
        a = rng.normal(loc=0, scale=1, size=20)
        b = rng.normal(loc=0.5, scale=1, size=20)
        result_strict = welch_ttest(a, b, alpha=0.001)
        result_lenient = welch_ttest(a, b, alpha=0.5)
        assert result_strict.alpha == 0.001
        assert result_lenient.alpha == 0.5

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            welch_ttest(np.array([1.0]), np.array([1.0, 2.0]))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="Alpha must be in"):
            welch_ttest(np.array([1.0, 2.0]), np.array([3.0, 4.0]), alpha=0.0)

    def test_degrees_of_freedom_positive(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = welch_ttest(a, b)
        assert result.df > 0

    def test_symmetric(self):
        """Swapping groups should negate t-statistic but preserve p-value."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        r1 = welch_ttest(a, b)
        r2 = welch_ttest(b, a)
        assert abs(r1.t_statistic + r2.t_statistic) < 1e-10
        assert abs(r1.p_value - r2.p_value) < 1e-10
