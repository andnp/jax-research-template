"""Small tests for research_analysis.bootstrap — CI computation."""

import numpy as np
import pytest
from research_analysis.bootstrap import BootstrapCI, bootstrap_ci


def _reference_bootstrap_ci(
    data: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> BootstrapCI:
    indices = rng.integers(0, data.shape[0], size=(n_resamples, data.shape[0]))
    boot_means = data[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return BootstrapCI(
        mean=data.mean(axis=0),
        ci_low=np.percentile(boot_means, 100 * alpha / 2, axis=0),
        ci_high=np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0),
        confidence=confidence,
    )


class TestBootstrapCI:
    def test_basic_shape(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(10, 50))
        result = bootstrap_ci(data, rng=rng)
        assert isinstance(result, BootstrapCI)
        assert result.mean.shape == (50,)
        assert result.ci_low.shape == (50,)
        assert result.ci_high.shape == (50,)
        assert result.confidence == 0.95

    def test_1d_input(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20,))
        result = bootstrap_ci(data, rng=rng)
        assert result.mean.shape == ()
        assert result.ci_low.shape == ()
        assert result.ci_high.shape == ()

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=5.0, size=(30, 10))
        result = bootstrap_ci(data, rng=rng)
        assert np.all(result.ci_low <= result.mean)
        assert np.all(result.ci_high >= result.mean)

    def test_wider_ci_with_more_variance(self):
        rng = np.random.default_rng(42)
        low_var = rng.normal(loc=0, scale=0.1, size=(10, 20))
        high_var = rng.normal(loc=0, scale=10.0, size=(10, 20))
        ci_low = bootstrap_ci(low_var, rng=np.random.default_rng(42))
        ci_high = bootstrap_ci(high_var, rng=np.random.default_rng(42))
        width_low = (ci_low.ci_high - ci_low.ci_low).mean()
        width_high = (ci_high.ci_high - ci_high.ci_low).mean()
        assert width_high > width_low

    def test_custom_confidence(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(10, 20))
        ci_90 = bootstrap_ci(data, confidence=0.90, rng=rng)
        ci_99 = bootstrap_ci(data, confidence=0.99, rng=np.random.default_rng(42))
        assert ci_90.confidence == 0.90
        assert ci_99.confidence == 0.99
        # 99% CI should be wider than 90%
        width_90 = (ci_90.ci_high - ci_90.ci_low).mean()
        width_99 = (ci_99.ci_high - ci_99.ci_low).mean()
        assert width_99 > width_90

    def test_too_few_seeds_raises(self):
        with pytest.raises(ValueError, match="at least 2 seeds"):
            bootstrap_ci(np.array([[1.0, 2.0]]))

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="Confidence must be in"):
            bootstrap_ci(np.ones((3, 5)), confidence=1.5)

    def test_matches_reference_numpy_bootstrap(self):
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 6.0, 9.0],
                [4.0, 8.0, 12.0],
            ]
        )
        confidence = 0.9
        n_resamples = 512

        result = bootstrap_ci(
            data,
            confidence=confidence,
            n_resamples=n_resamples,
            rng=np.random.default_rng(1234),
        )
        expected = _reference_bootstrap_ci(
            data,
            confidence=confidence,
            n_resamples=n_resamples,
            rng=np.random.default_rng(1234),
        )

        np.testing.assert_allclose(result.mean, expected.mean)
        np.testing.assert_allclose(result.ci_low, expected.ci_low)
        np.testing.assert_allclose(result.ci_high, expected.ci_high)
