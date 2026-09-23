"""Small tests for research_analysis.bootstrap — CI computation."""

import numpy as np
import pytest
from research_analysis.bootstrap import (
    BootstrapCI,
    PercentileBootstrapCI,
    ScalarBootstrapCI,
    bootstrap_ci,
    bootstrap_difference_ci,
    bootstrap_percentile_ci,
    bootstrap_scalar_ci,
    circular_moving_block_resample,
    resolve_block_size,
)


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


def test_resolve_block_size_uses_cube_root_default() -> None:
    """
    Verify the default moving-block length follows the cube-root rule.
    """
    assert resolve_block_size(1) == 1
    assert resolve_block_size(8) == 2
    assert resolve_block_size(27) == 3
    assert resolve_block_size(20, block_size=4) == 4
    assert resolve_block_size(20, block_size=40) == 40


def test_resolve_block_size_rejects_invalid_values() -> None:
    """
    Verify block-size inputs require positive integer configuration values.
    """
    with pytest.raises(ValueError):
        resolve_block_size(0)
    with pytest.raises(ValueError):
        resolve_block_size(4, block_size=0)
    with pytest.raises(ValueError):
        resolve_block_size(4, block_size=True)


def test_circular_resample_preserves_centered_circular_blocks() -> None:
    """
    Verify sampled blocks remain contiguous across the circular boundary.
    """
    residuals = np.arange(6, dtype=np.float64)
    result = circular_moving_block_resample(
        residuals,
        block_size=4,
        rng=np.random.default_rng(7),
    )
    indices = np.rint(result + residuals.mean()).astype(np.int64)

    assert result.shape == residuals.shape
    assert np.all(np.isin(result, residuals - residuals.mean()))
    assert np.all(np.diff(indices[:4]) % residuals.size == 1)
    assert np.all(np.diff(indices[4:]) % residuals.size == 1)


def test_circular_resample_is_reproducible_and_centered() -> None:
    """
    Verify seeded resampling is reproducible and centers its residual distribution.
    """
    residuals = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    draws = [
        circular_moving_block_resample(
            residuals,
            block_size=4,
            rng=np.random.default_rng(seed),
        )
        for seed in range(512)
    ]

    first = circular_moving_block_resample(residuals, rng=np.random.default_rng(7))
    second = circular_moving_block_resample(residuals, rng=np.random.default_rng(7))

    assert np.allclose(first, second)
    assert np.mean(np.concatenate(draws)) == pytest.approx(0.0, abs=0.1)


def test_circular_resample_rejects_invalid_inputs() -> None:
    """
    Verify empty, non-one-dimensional, and non-finite residuals are rejected.
    """
    with pytest.raises(ValueError):
        circular_moving_block_resample(np.asarray([]), rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        circular_moving_block_resample(np.ones((2, 2)), rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        circular_moving_block_resample(np.asarray([1.0, np.nan]), rng=np.random.default_rng(1))


def test_percentile_bootstrap_matches_numpy_mean_reference() -> None:
    """
    Verify the mean percentile interval matches direct NumPy resampling.
    """
    values = np.asarray([1.0, 2.0, 4.0, 7.0])
    confidence = 0.9
    n_resamples = 128
    result = bootstrap_percentile_ci(
        values,
        confidence=confidence,
        n_resamples=n_resamples,
        rng=np.random.default_rng(1234),
    )
    rng = np.random.default_rng(1234)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    estimates = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0

    assert isinstance(result, PercentileBootstrapCI)
    assert result.statistic == "mean"
    assert result.estimate == pytest.approx(values.mean())
    assert result.ci_low == pytest.approx(np.quantile(estimates, alpha))
    assert result.ci_high == pytest.approx(np.quantile(estimates, 1.0 - alpha))


def test_percentile_bootstrap_supports_interquartile_mean() -> None:
    """
    Verify percentile resampling supports the shared interquartile mean statistic.
    """
    result = bootstrap_percentile_ci(
        np.asarray([1.0, 2.0, 4.0, 7.0]),
        statistic="interquartile_mean",
        n_resamples=128,
        rng=np.random.default_rng(42),
    )

    assert result.statistic == "interquartile_mean"
    assert result.estimate == pytest.approx(3.0)
    assert result.ci_low <= result.estimate <= result.ci_high


def test_percentile_bootstrap_rejects_invalid_inputs() -> None:
    """
    Verify percentile bootstrap validation rejects underpowered and invalid samples.
    """
    with pytest.raises(ValueError):
        bootstrap_percentile_ci(np.asarray([1.0]))
    with pytest.raises(ValueError):
        bootstrap_percentile_ci(np.asarray([1.0, np.nan]))
    with pytest.raises(ValueError):
        bootstrap_percentile_ci(np.ones(3), confidence=1.0)
    with pytest.raises(ValueError):
        bootstrap_percentile_ci(np.ones(3), n_resamples=1)


class TestBootstrapCI:
    def test_basic_shape(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(10, 50))
        result = bootstrap_ci(data, rng=rng)
        assert isinstance(result, BootstrapCI)
        assert result.mean.shape == (50,)
        assert result.ci_low.shape == (50,)
        assert result.ci_high.shape == (50,)
        assert result.confidence == 0.95

    def test_1d_input(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20,))
        result = bootstrap_ci(data, rng=rng)
        assert result.mean.shape == ()
        assert result.ci_low.shape == ()
        assert result.ci_high.shape == ()

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(loc=5.0, size=(30, 10))
        result = bootstrap_ci(data, rng=rng)
        assert np.all(result.ci_low <= result.mean)
        assert np.all(result.ci_high >= result.mean)

    def test_wider_ci_with_more_variance(self) -> None:
        rng = np.random.default_rng(42)
        low_var = rng.normal(loc=0, scale=0.1, size=(10, 20))
        high_var = rng.normal(loc=0, scale=10.0, size=(10, 20))
        ci_low = bootstrap_ci(low_var, rng=np.random.default_rng(42))
        ci_high = bootstrap_ci(high_var, rng=np.random.default_rng(42))
        width_low = (ci_low.ci_high - ci_low.ci_low).mean()
        width_high = (ci_high.ci_high - ci_high.ci_low).mean()
        assert width_high > width_low

    def test_custom_confidence(self) -> None:
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

    def test_too_few_seeds_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 seeds"):
            bootstrap_ci(np.array([[1.0, 2.0]]))

    def test_non_finite_values_raise_naming_the_seed_rows(self) -> None:
        """A diverged seed must not silently turn the interval into NaN."""
        data = np.ones((3, 4))
        data[1, 2] = np.nan
        with pytest.raises(ValueError, match=r"rows \[1\]"):
            bootstrap_ci(data)

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="Confidence must be in"):
            bootstrap_ci(np.ones((3, 5)), confidence=1.5)

    def test_matches_reference_numpy_bootstrap(self) -> None:
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


def test_paired_difference_resamples_runs_jointly() -> None:
    """A paired interval resamples each seed's pair together, matching a NumPy reference."""
    a = np.array([1.0, 2.0, 4.0, 7.0, 9.0])
    b = np.array([0.0, 1.0, 2.0, 4.0, 5.0])
    result = bootstrap_difference_ci(
        a, b, pairing="paired", confidence=0.90, n_resamples=128, rng=np.random.default_rng(42)
    )
    indices = np.random.default_rng(42).integers(0, a.size, size=(128, a.size))
    replicates = a[indices].mean(axis=1) - b[indices].mean(axis=1)

    assert result.estimate == pytest.approx(2.2)
    assert result.ci_low == pytest.approx(np.quantile(replicates, 0.05))
    assert result.ci_high == pytest.approx(np.quantile(replicates, 0.95))
    assert result.confidence == 0.90
    assert result.n_resamples == 128
    assert result.pairing == "paired"


def test_independent_difference_resamples_each_arm_separately() -> None:
    """An independent interval draws each arm's resample on its own, so arms may differ in size."""
    a = np.array([1.0, 2.0, 4.0, 7.0])
    b = np.array([0.0, 1.0, 2.0])
    result = bootstrap_difference_ci(a, b, n_resamples=64, rng=np.random.default_rng(3))
    rng = np.random.default_rng(3)
    replicates = a[rng.integers(0, 4, size=(64, 4))].mean(axis=1) - b[rng.integers(0, 3, size=(64, 3))].mean(axis=1)

    assert result.ci_low == pytest.approx(np.quantile(replicates, 0.025))
    assert result.ci_high == pytest.approx(np.quantile(replicates, 0.975))


def test_difference_bootstrap_rejects_invalid_inputs() -> None:
    """Difference intervals reject underpowered, misaligned, and non-finite data."""
    with pytest.raises(ValueError, match="at least 3"):
        bootstrap_difference_ci(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="equal lengths"):
        bootstrap_difference_ci(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 1.0]),
            pairing="paired",
        )
    with pytest.raises(ValueError, match="finite"):
        bootstrap_difference_ci(
            np.array([1.0, np.nan, 3.0]),
            np.array([0.0, 1.0, 2.0]),
        )


def test_constant_paired_difference_is_a_point() -> None:
    """A constant paired difference resamples to itself, so its interval collapses to it."""
    result = bootstrap_difference_ci(
        np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]), pairing="paired", rng=np.random.default_rng(0)
    )

    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)


def test_scalar_bootstrap_matches_numpy_percentile_reference() -> None:
    """Scalar intervals are the percentile bounds of resampled seed statistics."""
    values = np.array([1.0, 2.0, 4.0, 7.0, 9.0])
    result = bootstrap_scalar_ci(values, confidence=0.90, n_resamples=128, rng=np.random.default_rng(42))
    indices = np.random.default_rng(42).integers(0, values.size, size=(128, values.size))
    replicates = values[indices].mean(axis=1)

    assert isinstance(result, ScalarBootstrapCI)
    assert result.estimate == pytest.approx(4.6)
    assert result.ci_low == pytest.approx(np.quantile(replicates, 0.05))
    assert result.ci_high == pytest.approx(np.quantile(replicates, 0.95))
    assert result.confidence == 0.90


def test_scalar_bootstrap_of_a_constant_sample_is_a_point() -> None:
    """A constant sample resamples to itself, so its interval collapses to the value."""
    result = bootstrap_scalar_ci(np.array([4.0, 4.0, 4.0]), rng=np.random.default_rng(0))

    assert result.ci_low == pytest.approx(4.0)
    assert result.ci_high == pytest.approx(4.0)


def test_scalar_bootstrap_supports_interquartile_mean() -> None:
    """The IQM estimand trims each resample the same way as the point estimate."""
    values = np.array([1.0, 2.0, 4.0, 7.0])
    result = bootstrap_scalar_ci(values, estimand="interquartile_mean", n_resamples=64, rng=np.random.default_rng(5))
    indices = np.random.default_rng(5).integers(0, values.size, size=(64, values.size))
    replicates = np.sort(values[indices], axis=1)[:, 1:3].mean(axis=1)

    assert result.estimand == "interquartile_mean"
    assert result.estimate == pytest.approx(3.0)
    assert result.ci_low == pytest.approx(np.quantile(replicates, 0.025))
    assert result.ci_high == pytest.approx(np.quantile(replicates, 0.975))


def test_scalar_bootstrap_accepts_two_runs_only_when_declared() -> None:
    """Two runs need an explicit min_samples=2; one run never forms an interval."""
    values = np.array([1.0, 3.0])
    with pytest.raises(ValueError, match="at least 3"):
        bootstrap_scalar_ci(values, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_scalar_ci(values[:1], min_samples=2, rng=np.random.default_rng(0))

    result = bootstrap_scalar_ci(values, min_samples=2, rng=np.random.default_rng(0))

    assert 1.0 <= result.ci_low <= result.ci_high <= 3.0
