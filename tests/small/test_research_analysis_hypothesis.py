"""Small tests for research_analysis.hypothesis."""

import numpy as np
import pytest
from research_analysis.hypothesis import (
    MannWhitneyResult,
    WelchResult,
    mann_whitney_u_test,
    welch_ttest,
)


class TestWelchTtest:
    def test_identical_groups_not_significant(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = welch_ttest(a, b)
        assert isinstance(result, WelchResult)
        assert result.t_statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert not result.significant

    def test_different_groups_significant(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = welch_ttest(a, b)
        assert result.significant
        assert result.p_value < 0.05
        assert result.t_statistic < 0.0

    def test_means_correct(self):
        a = np.array([1.0, 2.0, 7.0])
        b = np.array([4.0, 5.0, 6.0])
        result = welch_ttest(a, b)
        assert result.mean_a == pytest.approx(a.mean())
        assert result.mean_b == pytest.approx(b.mean())

    def test_custom_alpha(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        result_strict = welch_ttest(a, b, alpha=0.001)
        result_lenient = welch_ttest(a, b, alpha=0.5)
        assert result_strict.alpha == 0.001
        assert result_lenient.alpha == 0.5
        assert not result_strict.significant
        assert result_lenient.significant

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
        assert result.df > 0.0

    def test_swap_groups_preserves_p_value_and_flips_statistic(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        first = welch_ttest(a, b)
        second = welch_ttest(b, a)
        assert first.t_statistic == pytest.approx(-second.t_statistic)
        assert first.p_value == pytest.approx(second.p_value)


class TestMannWhitneyUTest:
    def test_identical_groups_not_significant(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = mann_whitney_u_test(a, b)
        assert isinstance(result, MannWhitneyResult)
        assert result.p_value == pytest.approx(1.0)
        assert result.rank_biserial_correlation == pytest.approx(0.0)
        assert not result.significant

    def test_different_groups_significant(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = mann_whitney_u_test(a, b)
        assert result.significant
        assert result.p_value < 0.05
        assert result.u_statistic == pytest.approx(0.0)
        assert result.rank_biserial_correlation == pytest.approx(-1.0)

    def test_medians_correct(self):
        a = np.array([1.0, 2.0, 7.0])
        b = np.array([4.0, 5.0, 6.0])
        result = mann_whitney_u_test(a, b)
        assert result.median_a == pytest.approx(2.0)
        assert result.median_b == pytest.approx(5.0)

    def test_custom_alpha(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result_strict = mann_whitney_u_test(a, b, alpha=0.001)
        result_lenient = mann_whitney_u_test(a, b, alpha=0.5)
        assert result_strict.alpha == 0.001
        assert result_lenient.alpha == 0.5
        assert not result_strict.significant
        assert result_lenient.significant

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            mann_whitney_u_test(np.array([]), np.array([1.0, 2.0]))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="Alpha must be in"):
            mann_whitney_u_test(np.array([1.0, 2.0]), np.array([3.0, 4.0]), alpha=0.0)

    def test_swap_groups_preserves_p_value_and_flips_effect(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        first = mann_whitney_u_test(a, b)
        second = mann_whitney_u_test(b, a)
        assert first.p_value == pytest.approx(second.p_value)
        assert first.rank_biserial_correlation == pytest.approx(
            -second.rank_biserial_correlation
        )
        assert first.u_statistic + second.u_statistic == pytest.approx(len(a) * len(b))

    def test_ties_are_handled_without_false_positive(self):
        a = np.array([1.0, 1.0, 2.0, 2.0])
        b = np.array([1.0, 2.0, 2.0, 3.0])
        result = mann_whitney_u_test(a, b)
        assert 0.0 <= result.p_value <= 1.0
        assert not result.significant
