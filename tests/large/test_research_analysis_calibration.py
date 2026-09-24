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
from research_analysis.bootstrap import bootstrap_difference_ci, bootstrap_scalar_ci
from scipy import stats

CONFIDENCE = 0.95
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


def mean_ci_coverage(distribution: RunDistribution, n: int, replicates: int, rng: np.random.Generator):
    hits = 0
    for _ in range(replicates):
        ci = bootstrap_scalar_ci(distribution.sample(rng, n), n_resamples=N_RESAMPLES, rng=rng)
        hits += ci.ci_low <= distribution.mean <= ci.ci_high
    return hits / replicates


def paired_samples(distribution: RunDistribution, n: int, shift: float, rng: np.random.Generator):
    seed_effect = rng.standard_normal(n)
    return seed_effect + distribution.sample(rng, n) + shift, seed_effect + distribution.sample(rng, n)


def difference_ci_coverage(distribution: RunDistribution, n: int, replicates: int, rng: np.random.Generator):
    shift, hits = 0.5, 0
    for _ in range(replicates):
        values_a, values_b = paired_samples(distribution, n, shift, rng)
        ci = bootstrap_difference_ci(values_a, values_b, pairing="paired", n_resamples=N_RESAMPLES, rng=rng)
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
