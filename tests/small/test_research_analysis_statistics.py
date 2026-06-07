"""Small tests for research_analysis.statistics — Tolerance intervals & median run selection."""

import numpy as np
import pytest
from research_analysis.statistics import (
    ToleranceInterval,
    pointwise_tolerance_interval,
    select_median_run_index,
    tolerance_interval_confidence,
    tolerance_interval_order_indices,
)


class TestToleranceInterval:
    def test_tolerance_interval_confidence(self) -> None:
        # For n = 4, coverage = 0.90
        # If rank_span = 3 (e.g. index 0 and index 3, covering index 1 and 2):
        # successes can be 0, 1, or 2 elements outside the interval.
        conf = tolerance_interval_confidence(4, coverage=0.90, rank_span=3)
        # Standard binomial: sum_{f=0}^{2} comb(4, f) * 0.9^f * 0.1^(4-f)
        # comb(4,0)*0.1^4 + comb(4,1)*0.9^1*0.1^3 + comb(4,2)*0.9^2*0.1^2
        # = 1 * 0.0001 + 4 * 0.0009 + 6 * 0.0081 = 0.0001 + 0.0036 + 0.0486 = 0.0523
        assert pytest.approx(conf) == 0.0523

    def test_tolerance_interval_order_indices(self) -> None:
        # Let's verify standard values. For a large number of seeds, say 100, we should be able to get a 95% confidence, 90% coverage TI.
        lower_idx, upper_idx = tolerance_interval_order_indices(100, confidence=0.95, coverage=0.90)
        assert lower_idx >= 0
        assert upper_idx < 100
        assert upper_idx > lower_idx

    def test_insufficient_samples_raises(self) -> None:
        # For very small sample sizes (e.g. 5 seeds), we cannot achieve 95% confidence for 90% coverage.
        with pytest.raises(ValueError, match="Insufficient samples"):
            tolerance_interval_order_indices(5, confidence=0.95, coverage=0.90)

    def test_invalid_parameters_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            tolerance_interval_order_indices(10, confidence=-0.1, coverage=0.9)
        with pytest.raises(ValueError, match="coverage must be in"):
            tolerance_interval_order_indices(10, confidence=0.95, coverage=1.2)
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            tolerance_interval_order_indices(1, confidence=0.95, coverage=0.90)

    def test_pointwise_tolerance_interval_2d(self) -> None:
        # Let's create a dataset of 100 seeds and 50 steps
        rng = np.random.default_rng(42)
        data = rng.normal(loc=10.0, scale=2.0, size=(100, 50))
        ti = pointwise_tolerance_interval(data, confidence=0.95, coverage=0.90)
        
        assert isinstance(ti, ToleranceInterval)
        assert ti.ci_low.shape == (50,)
        assert ti.ci_high.shape == (50,)
        assert ti.confidence == 0.95
        assert ti.coverage == 0.90
        assert np.all(ti.ci_low <= ti.ci_high)

    def test_pointwise_tolerance_interval_1d(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(loc=10.0, scale=2.0, size=(100,))
        ti = pointwise_tolerance_interval(data, confidence=0.95, coverage=0.90)
        
        assert isinstance(ti.ci_low, np.float64)
        assert isinstance(ti.ci_high, np.float64)
        assert ti.ci_low <= ti.ci_high


class TestSelectMedianRunIndex:
    def test_select_median_run_index_1d(self) -> None:
        data = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
        # median of [1.0, 2.0, 10.0, 11.0, 12.0] is 10.0 (index 2)
        idx = select_median_run_index(data)
        assert idx == 2

    def test_select_median_run_index_2d(self) -> None:
        # 3 seeds, 5 steps each.
        # Seed 0: mean = 2.0
        # Seed 1: mean = 5.0
        # Seed 2: mean = 9.0
        # Median mean is 5.0, so seed 1 should be selected.
        data = np.array([
            [1.0, 2.0, 2.0, 2.0, 3.0], # mean = 2.0
            [4.0, 5.0, 5.0, 5.0, 6.0], # mean = 5.0
            [8.0, 9.0, 9.0, 9.0, 10.0] # mean = 9.0
        ])
        idx = select_median_run_index(data)
        assert idx == 1

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError, match="Input data must not be empty"):
            select_median_run_index(np.array([]))

    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 1-D or 2-D array"):
            select_median_run_index(np.ones((2, 2, 2)))
