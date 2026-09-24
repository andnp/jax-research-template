"""Behavior tests for permutation inference."""

from typing import Literal

import numpy as np
import pytest
from research_analysis.permutation import PermutationResult, run_permutation_test

Pairing = Literal["paired", "independent"]


def test_paired_permutation_preserves_declared_design() -> None:
    """Paired samples report the mean difference and paired permutation design."""
    result = run_permutation_test(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        pairing="paired",
        n_resamples=256,
        rng=np.random.default_rng(4),
    )

    assert isinstance(result, PermutationResult)
    assert result.statistic == pytest.approx(1.0)
    assert result.pairing == "paired"
    assert result.estimand == "mean"


def test_independent_permutation_supports_median_estimand() -> None:
    """Independent samples can test a median difference without normality checks."""
    result = run_permutation_test(
        np.array([1.0, 2.0, 100.0]),
        np.array([4.0, 5.0, 6.0]),
        estimand="median",
        pairing="independent",
        n_resamples=128,
        rng=np.random.default_rng(8),
    )

    assert result.statistic == pytest.approx(-3.0)
    assert result.estimand == "median"
    assert result.pairing == "independent"


def test_paired_median_tests_the_median_of_per_seed_differences() -> None:
    """A paired median statistic is median(a - b), which uses the pairing, not median(a) - median(b)."""
    result = run_permutation_test(
        np.array([1.0, 2.0, 30.0]),
        np.array([0.0, 5.0, 6.0]),
        estimand="median",
        pairing="paired",
        n_resamples=8,
        rng=np.random.default_rng(0),
    )

    assert result.statistic == pytest.approx(1.0)


def test_permutation_is_reproducible_with_same_rng_seed() -> None:
    """The configured random stream makes finite permutation results reproducible."""
    values_a = np.array([0.0, 1.0, 3.0, 4.0])
    values_b = np.array([0.0, 2.0, 2.0, 5.0])

    first = run_permutation_test(values_a, values_b, n_resamples=256, rng=np.random.default_rng(9))
    second = run_permutation_test(values_a, values_b, n_resamples=256, rng=np.random.default_rng(9))

    assert first.p_value == second.p_value


@pytest.mark.parametrize(
    ("values_a", "values_b", "pairing", "message"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "paired", "at least 3"),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), "paired", "equal lengths"),
        (np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]), "independent", "finite"),
    ],
)
def test_rejects_invalid_permutation_samples(
    values_a: np.ndarray,
    values_b: np.ndarray,
    pairing: Pairing,
    message: str,
) -> None:
    """Underpowered, misaligned, and non-finite samples fail explicitly."""
    with pytest.raises(ValueError, match=message):
        run_permutation_test(values_a, values_b, pairing=pairing, rng=np.random.default_rng(0))
