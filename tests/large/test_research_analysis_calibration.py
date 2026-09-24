"""Monte-Carlo calibration of research_analysis intervals and tests against known ground truth.

Every measured rate is a proportion over R independent replicates, so its Monte-Carlo standard
error at a reference rate p0 is sqrt(p0 * (1 - p0) / R); bands are three standard errors wide.
The run distributions follow the handbook's warning that RL performance is skewed, bimodal, or
long-tailed with rare catastrophic runs (Patterson et al. 2024, sections 2.3 and 2.6).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from research_analysis.bootstrap import bootstrap_ci, bootstrap_difference_ci, bootstrap_scalar_ci
from research_analysis.comparison import Estimand, IntervalMethod
from research_analysis.multiplicity import bonferroni_confidence, holm_step_down
from research_analysis.permutation import run_permutation_test
from research_analysis.repeated_blocks import block_stratified_permutation_test
from research_analysis.statistics import (
    pointwise_tolerance_interval,
    tolerance_interval_confidence,
    tolerance_interval_order_indices,
)
from scipy import stats

CONFIDENCE = 0.95
ALPHA = 0.05
N_RESAMPLES = 1000

type Sampler = Callable[[np.random.Generator, int | tuple[int, ...]], NDArray[np.float64]]
type Cdf = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class RunDistribution:
    name: str
    sample: Sampler
    mean: float
    cdf: Cdf


def normal_mixture(name: str, components: tuple[tuple[float, float, float], ...]):
    weights, locs, scales = (np.array(column) for column in zip(*components, strict=True))

    def sample(rng: np.random.Generator, size: int | tuple[int, ...]):
        component = rng.choice(len(weights), size=size, p=weights)
        return locs[component] + scales[component] * rng.standard_normal(size)

    def cdf(values: NDArray[np.float64]):
        return np.sum(weights * stats.norm.cdf((values[..., np.newaxis] - locs) / scales), axis=-1)

    return RunDistribution(name, sample, float(np.sum(weights * locs)), cdf)


def reflected_lognormal(name: str, sigma: float):
    def sample(rng: np.random.Generator, size: int | tuple[int, ...]):
        return -rng.lognormal(0.0, sigma, size)

    def cdf(values: NDArray[np.float64]):
        magnitude = np.log(np.maximum(-values, np.finfo(np.float64).tiny)) / sigma
        return np.where(values < 0.0, stats.norm.sf(magnitude), 1.0)

    return RunDistribution(name, sample, -math.exp(sigma**2 / 2.0), cdf)


NORMAL = normal_mixture("normal", ((1.0, 0.0, 1.0),))
MILD_SKEW = reflected_lognormal("mild_skew", 0.25)
BIMODAL = normal_mixture("bimodal", ((0.3, 0.0, 0.1), (0.7, 1.0, 0.1)))
HEAVY_SKEW = reflected_lognormal("heavy_skew", 0.8)
CATASTROPHE_RATE = 0.05
CATASTROPHIC = normal_mixture("catastrophic", ((1.0 - CATASTROPHE_RATE, 1.0, 0.1), (CATASTROPHE_RATE, -10.0, 1.0)))


def mc_band(reference: float, replicates: int):
    return 3.0 * math.sqrt(reference * (1.0 - reference) / replicates)


def gaussian_percentile_coverage(n: int):
    # The percentile bootstrap of a Gaussian mean is z times the plug-in SE, whose (n - 1) / n
    # variance makes it cover P(|t_{n-1}| <= z * sqrt((n - 1) / n)), below nominal at small n.
    z = stats.norm.ppf(0.5 + CONFIDENCE / 2.0)
    return float(2.0 * stats.t.cdf(z * math.sqrt((n - 1) / n), df=n - 1) - 1.0)


def mean_ci_coverage(
    distribution: RunDistribution,
    n: int,
    replicates: int,
    rng: np.random.Generator,
    interval: IntervalMethod = "percentile",
):
    hits = 0
    for _ in range(replicates):
        ci = bootstrap_scalar_ci(distribution.sample(rng, n), n_resamples=N_RESAMPLES, interval=interval, rng=rng)
        hits += ci.ci_low <= distribution.mean <= ci.ci_high
    return hits / replicates


def paired_samples(distribution: RunDistribution, n: int, shift: float, rng: np.random.Generator):
    seed_effect = rng.standard_normal(n)
    return seed_effect + distribution.sample(rng, n) + shift, seed_effect + distribution.sample(rng, n)


def difference_ci_coverage(
    distribution: RunDistribution,
    n: int,
    replicates: int,
    rng: np.random.Generator,
    interval: IntervalMethod = "percentile",
):
    shift, hits = 0.5, 0
    for _ in range(replicates):
        values_a, values_b = paired_samples(distribution, n, shift, rng)
        ci = bootstrap_difference_ci(
            values_a, values_b, pairing="paired", n_resamples=N_RESAMPLES, interval=interval, rng=rng
        )
        hits += ci.ci_low <= shift <= ci.ci_high
    return hits / replicates


# ── Percentile bootstrap CI of the mean ──────────────────────────────────────


@pytest.mark.parametrize("distribution", [NORMAL, MILD_SKEW, BIMODAL], ids=lambda d: d.name)
def test_mean_ci_covers_near_nominal_at_thirty_runs(distribution: RunDistribution) -> None:
    """The mean CI is calibrated at n=30 for non-pathological run distributions.

    The floor is the Gaussian percentile-bootstrap coverage (below nominal even for normal data)
    minus 1.5 points of allowance for shape, minus the Monte-Carlo band.
    """
    replicates = 4000
    coverage = mean_ci_coverage(distribution, 30, replicates, np.random.default_rng(0))

    band = mc_band(CONFIDENCE, replicates)
    assert gaussian_percentile_coverage(30) - 0.015 - band <= coverage <= CONFIDENCE + band, coverage


@pytest.mark.parametrize("distribution", [NORMAL, MILD_SKEW, BIMODAL], ids=lambda d: d.name)
def test_mean_ci_under_covers_at_ten_runs(distribution: RunDistribution) -> None:
    """At n=10 the mean CI under-covers by several points even for well-behaved runs.

    Records current behaviour, so a change of interval method shows up as a deliberate update here.
    """
    replicates = 4000
    coverage = mean_ci_coverage(distribution, 10, replicates, np.random.default_rng(0))

    assert 0.85 <= coverage <= gaussian_percentile_coverage(10) + mc_band(CONFIDENCE, replicates), coverage


@pytest.mark.parametrize("n", [10, 30])
def test_mean_ci_under_covers_heavily_skewed_runs(n: int) -> None:
    """A heavily skewed run distribution costs coverage beyond the Gaussian shortfall.

    Records current behaviour, so a change of interval method shows up as a deliberate update here.
    """
    replicates = 4000
    coverage = mean_ci_coverage(HEAVY_SKEW, n, replicates, np.random.default_rng(0))

    assert coverage < gaussian_percentile_coverage(n) - mc_band(CONFIDENCE, replicates), coverage


@pytest.mark.parametrize("n", [10, 30])
def test_mean_ci_misses_rare_catastrophes_in_small_samples(n: int) -> None:
    """With 5% catastrophic runs, a sample with none cannot cover the mean (handbook 2.6, PuddleWorld).

    Coverage is therefore capped by the chance of seeing a catastrophe, 1 - 0.95**n.
    Records current behaviour, so a change of interval method shows up as a deliberate update here.
    """
    replicates = 4000
    coverage = mean_ci_coverage(CATASTROPHIC, n, replicates, np.random.default_rng(0))

    cap = 1.0 - (1.0 - CATASTROPHE_RATE) ** n
    assert coverage <= cap + mc_band(cap, replicates), coverage


@pytest.mark.parametrize("distribution", [NORMAL, BIMODAL], ids=lambda d: d.name)
def test_pointwise_curve_ci_covers_each_step_at_thirty_runs(distribution: RunDistribution) -> None:
    """Each step of a learning-curve CI is as calibrated as the scalar mean CI.

    Every column is an independent replicate; steps share resample indices as real curves do,
    which correlates their coverage only weakly because the data are independent.
    """
    steps = 4000
    curves = distribution.sample(np.random.default_rng(0), (30, steps))

    ci = bootstrap_ci(curves, confidence=CONFIDENCE, n_resamples=N_RESAMPLES, rng=np.random.default_rng(1))
    coverage = float(np.mean((ci.ci_low <= distribution.mean) & (distribution.mean <= ci.ci_high)))

    band = mc_band(CONFIDENCE, steps)
    assert gaussian_percentile_coverage(30) - 0.015 - band <= coverage <= CONFIDENCE + band, coverage


# ── Paired percentile bootstrap CI of a difference ──────────────────────────


@pytest.mark.parametrize("distribution", [NORMAL, MILD_SKEW, BIMODAL], ids=lambda d: d.name)
def test_difference_ci_covers_near_nominal_at_thirty_runs(distribution: RunDistribution) -> None:
    """The paired difference CI, the bakeoff headline, is calibrated at n=30 with a shared seed effect."""
    replicates = 4000
    coverage = difference_ci_coverage(distribution, 30, replicates, np.random.default_rng(0))

    band = mc_band(CONFIDENCE, replicates)
    assert gaussian_percentile_coverage(30) - 0.015 - band <= coverage <= CONFIDENCE + band, coverage


@pytest.mark.parametrize("distribution", [NORMAL, MILD_SKEW, BIMODAL], ids=lambda d: d.name)
def test_difference_ci_under_covers_at_ten_runs(distribution: RunDistribution) -> None:
    """At n=10 the paired difference CI under-covers by several points.

    Records current behaviour, so a change of interval method shows up as a deliberate update here.
    """
    replicates = 4000
    coverage = difference_ci_coverage(distribution, 10, replicates, np.random.default_rng(0))

    assert 0.85 <= coverage <= gaussian_percentile_coverage(10) + mc_band(CONFIDENCE, replicates), coverage


def test_difference_ci_under_covers_catastrophic_runs_at_thirty() -> None:
    """Rare catastrophes in either arm still cost coverage at n=30.

    Records current behaviour, so a change of interval method shows up as a deliberate update here.
    """
    replicates = 4000
    coverage = difference_ci_coverage(CATASTROPHIC, 30, replicates, np.random.default_rng(0))

    assert coverage < gaussian_percentile_coverage(30) - mc_band(CONFIDENCE, replicates), coverage


# ── Distribution-free tolerance intervals (handbook 2.4, Appendix C) ─────────


@pytest.mark.parametrize("distribution", [NORMAL, HEAVY_SKEW, BIMODAL, CATASTROPHIC], ids=lambda d: d.name)
@pytest.mark.parametrize(
    ("n", "confidence", "proportion"),
    [(46, 0.95, 0.90), (60, 0.95, 0.90), (100, 0.95, 0.90), (200, 0.95, 0.90), (22, 0.90, 0.80)],
)
def test_tolerance_interval_contains_proportion_with_stated_confidence(
    distribution: RunDistribution, n: int, confidence: float, proportion: float
) -> None:
    """A tolerance interval holds at least the proportion of the population with its exact achieved confidence.

    For a continuous distribution the order-statistic interval's confidence is exact and at least nominal,
    whatever the shape, so the measured rate must match it rather than merely exceed it.
    """
    replicates = 20_000
    runs = distribution.sample(np.random.default_rng(0), (n, replicates))

    interval = pointwise_tolerance_interval(runs, confidence=confidence, coverage=proportion)
    content = distribution.cdf(np.asarray(interval.ci_high)) - distribution.cdf(np.asarray(interval.ci_low))
    achieved = float(np.mean(content >= proportion))

    lower, upper = tolerance_interval_order_indices(n, confidence=confidence, coverage=proportion)
    exact = tolerance_interval_confidence(n, coverage=proportion, rank_span=upper - lower)
    assert exact >= confidence
    assert abs(achieved - exact) <= mc_band(exact, replicates), (achieved, exact)


# ── Type-I error of permutation tests ─────────────────────────────────────


def paired_permutation_rejection_rate(
    distribution: RunDistribution, n: int, estimand: Estimand, replicates: int, rng: np.random.Generator
):
    # 2**10 resamples makes the n=10 test exact; larger n uses a randomized test.
    n_resamples = 2**n if n <= 10 else 999
    rejections = 0
    for _ in range(replicates):
        values_a, values_b = paired_samples(distribution, n, 0.0, rng)
        result = run_permutation_test(
            values_a, values_b, estimand=estimand, pairing="paired", n_resamples=n_resamples, rng=rng
        )
        rejections += result.p_value < ALPHA
    return rejections / replicates


@pytest.mark.parametrize(
    ("distribution", "n", "estimand"),
    [
        (NORMAL, 10, "mean"),
        (NORMAL, 30, "mean"),
        (CATASTROPHIC, 10, "mean"),
        (CATASTROPHIC, 30, "mean"),
        (NORMAL, 10, "median"),
        (CATASTROPHIC, 10, "median"),
    ],
    ids=lambda value: value.name if isinstance(value, RunDistribution) else str(value),
)
def test_paired_permutation_test_holds_its_size(distribution: RunDistribution, n: int, estimand: Estimand) -> None:
    """Under a true null with a shared seed effect, the paired test rejects at most alpha of the time.

    The paired median is the median of per-seed differences; its measured rates were 0.050 (normal)
    and 0.047 (catastrophic) at n=10, using the whole alpha budget.
    """
    replicates = 1000
    rate = paired_permutation_rejection_rate(distribution, n, estimand, replicates, np.random.default_rng(0))

    assert rate <= ALPHA + mc_band(ALPHA, replicates), rate


def test_paired_permutation_test_is_not_overly_conservative() -> None:
    """With enough runs for fine p-values, the paired test uses its whole alpha budget, so it keeps power."""
    replicates = 1000
    rate = paired_permutation_rejection_rate(NORMAL, 30, "mean", replicates, np.random.default_rng(1))

    assert rate >= ALPHA - mc_band(ALPHA, replicates), rate


def test_paired_median_test_gains_power_from_pairing() -> None:
    """The paired median test detects a shift hidden under a large shared seed effect.

    Seed SD 1, within-pair SD 0.3, shift 0.5, n=10: measured rejection 0.886, near the paired mean
    test's 0.917. Differencing medians instead of taking the median of differences rejected about a
    third of the time, because the seed effect it fails to cancel swamps the shift. The floor is the
    measured rate less its Monte-Carlo band.
    """
    replicates, n, shift = 1000, 10, 0.5
    rng = np.random.default_rng(0)

    rejections = 0
    for _ in range(replicates):
        seed_effect = rng.standard_normal(n)
        values_a = seed_effect + rng.normal(0.0, 0.3, n) + shift
        values_b = seed_effect + rng.normal(0.0, 0.3, n)
        result = run_permutation_test(
            values_a, values_b, estimand="median", pairing="paired", n_resamples=2**n, rng=rng
        )
        rejections += result.p_value < ALPHA

    rate = rejections / replicates
    assert rate >= 0.886 - mc_band(0.886, replicates), rate


@pytest.mark.parametrize(
    ("distribution", "missing_cells"), [(NORMAL, 0), (CATASTROPHIC, 3)], ids=["complete-normal", "missing-catastrophic"]
)
def test_block_stratified_permutation_test_holds_its_size(distribution: RunDistribution, missing_cells: int) -> None:
    """Under a true null with strong block effects, the omnibus test rejects at most alpha of the time.

    Missing cells land in distinct blocks so each keeps two observed conditions, the shape the bakeoff
    passes under blockwise missingness.
    """
    replicates, blocks, conditions = 1000, 10, 3
    rng = np.random.default_rng(0)

    rejections = 0
    for _ in range(replicates):
        scores = 3.0 * rng.standard_normal((blocks, 1)) + distribution.sample(rng, (blocks, conditions))
        scores[rng.choice(blocks, size=missing_cells, replace=False), rng.integers(0, conditions, missing_cells)] = np.nan
        rejections += block_stratified_permutation_test(scores, n_resamples=99, rng=rng).p_value < ALPHA

    rate = rejections / replicates
    assert rate <= ALPHA + mc_band(ALPHA, replicates), rate


# ── Multiplicity across a bakeoff comparison family ─────────────────────────


def test_holm_controls_family_wise_error_over_dependent_pairs() -> None:
    """Holm-adjusted p-values over every pair of four null arms reject anything at most alpha of the time.

    The six paired tests share arms and seeds, so their p-values are dependent, as in a bakeoff.
    """
    replicates, n, arms = 1000, 10, 4
    rng = np.random.default_rng(0)

    family_errors = 0
    for _ in range(replicates):
        seed_effect = rng.standard_normal(n)
        values = [seed_effect + NORMAL.sample(rng, n) for _ in range(arms)]
        p_values = [
            run_permutation_test(values[i], values[j], pairing="paired", n_resamples=2**n, rng=rng).p_value
            for i in range(arms)
            for j in range(i + 1, arms)
        ]
        family_errors += min(holm_step_down(p_values)) < ALPHA

    rate = family_errors / replicates
    assert rate <= ALPHA + mc_band(ALPHA, replicates), rate


def bonferroni_family_coverage(distribution: RunDistribution, n: int, replicates: int, rng: np.random.Generator):
    shifts = np.array([0.0, 0.3, 0.6, 0.9])
    pairs = [(i, j) for i in range(len(shifts)) for j in range(i + 1, len(shifts))]
    confidence = bonferroni_confidence(CONFIDENCE, len(pairs))

    hits = np.zeros((replicates, len(pairs)), dtype=bool)
    for replicate in range(replicates):
        seed_effect = rng.standard_normal(n)
        values = [seed_effect + distribution.sample(rng, n) + shift for shift in shifts]
        for column, (i, j) in enumerate(pairs):
            ci = bootstrap_difference_ci(
                values[i], values[j], pairing="paired", confidence=confidence, n_resamples=2000, rng=rng
            )
            hits[replicate, column] = ci.ci_low <= shifts[i] - shifts[j] <= ci.ci_high
    return float(hits.mean()), float(hits.all(axis=1).mean()), len(pairs)


@pytest.mark.parametrize("distribution", [NORMAL, BIMODAL], ids=lambda d: d.name)
@pytest.mark.parametrize(("n", "expected_low", "expected_high"), [(10, 0.72, 0.88), (30, 0.88, 0.935), (100, 0.935, 0.965)])
def test_bonferroni_family_coverage_approaches_nominal_only_with_many_runs(
    distribution: RunDistribution, n: int, expected_low: float, expected_high: float
) -> None:
    """Six Bonferroni-adjusted paired CIs cover every true difference jointly about as often as Bonferroni allows.

    The union bound on the measured per-interval coverage always holds. Jointly the family reaches its
    nominal 95% only near n=100: at n=10 and n=30 the percentile bootstrap under-covers at the extreme
    per-interval level, and those ranges record current behaviour, so a change of interval method shows
    up as a deliberate update here.
    """
    replicates = 2000
    marginal, family, family_size = bonferroni_family_coverage(distribution, n, replicates, np.random.default_rng(0))

    band = mc_band(CONFIDENCE, replicates)
    assert family >= 1.0 - family_size * (1.0 - marginal) - band, (family, marginal)
    assert expected_low <= family <= expected_high, family
