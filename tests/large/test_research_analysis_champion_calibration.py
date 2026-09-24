"""Monte-Carlo calibration of tuned-champion CIs against the best true configuration mean.

The grids and run distributions reproduce projects/hyper-champion-ci experiment 001, whose diary
found the hierarchical per-sample bootstrap the best-calibrated champion CI. A coverage rate over R
replicates has Monte-Carlo standard error sqrt(p * (1 - p) / R); comparing against the diary's
2000-replicate estimate adds its own variance, so bands are 3 * sqrt(p * (1 - p) * (1 / R + 1 / 2000)).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from research_analysis.tuning import hierarchical_bootstrap_ci, hierarchical_bootstrap_difference_ci

REPLICATES = 1000
N_RESAMPLES = 1000

type GridSampler = Callable[[np.random.Generator, NDArray[np.float64], int], NDArray[np.float64]]


def bimodal(rng: np.random.Generator, means: NDArray[np.float64], n: int):
    modes = np.where(rng.integers(0, 2, (len(means), n)) == 0, -1.5, 1.5)
    return means[:, np.newaxis] + modes + rng.standard_normal((len(means), n))


def long_tailed(rng: np.random.Generator, means: NDArray[np.float64], n: int):
    return means[:, np.newaxis] + rng.lognormal(0.0, 0.8, (len(means), n)) - math.exp(0.32)


def dominant_nuisance(rng: np.random.Generator, means: NDArray[np.float64], n: int):
    outlier = rng.random((len(means), n)) < 0.05
    offsets = np.where(outlier, 0.2 * 0.95 / 0.05, -0.2)
    return means[:, np.newaxis] + offsets + np.where(outlier, 2.0, 0.3) * rng.standard_normal((len(means), n))


def champion_coverage(sample: GridSampler, h: int, n: int, rng: np.random.Generator):
    means = np.linspace(0.0, 1.0, h)
    hits = 0
    for _ in range(REPLICATES):
        ci = hierarchical_bootstrap_ci(sample(rng, means, n), n, n, n_resamples=N_RESAMPLES, rng=rng)
        hits += ci.ci_low <= 1.0 <= ci.ci_high
    return hits / REPLICATES


def champion_difference_coverage(sample: GridSampler, h: int, n: int, rng: np.random.Generator):
    means, gap = np.linspace(0.0, 1.0, h), 0.5
    hits = 0
    for _ in range(REPLICATES):
        ci = hierarchical_bootstrap_difference_ci(
            sample(rng, means, n), sample(rng, means - gap, n), n, n, n_resamples=N_RESAMPLES, rng=rng
        )
        hits += ci.ci_low <= gap <= ci.ci_high
    return hits / REPLICATES


def reference_band(reference: float):
    return 3.0 * math.sqrt(reference * (1.0 - reference) * (1.0 / REPLICATES + 1.0 / 2000))


# hier_per_sample_coverage from projects/hyper-champion-ci/results/001-onestage-twostage-loocv/sweep_results.csv
DIARY_COVERAGE = [
    (bimodal, 3, 10, 0.9535),
    (bimodal, 3, 30, 0.949),
    (bimodal, 10, 10, 0.9525),
    (bimodal, 10, 30, 0.9735),
    (long_tailed, 3, 10, 0.9245),
    (long_tailed, 3, 30, 0.927),
    (long_tailed, 10, 10, 0.995),
    (long_tailed, 10, 30, 0.989),
    (dominant_nuisance, 3, 10, 0.8235),
    (dominant_nuisance, 3, 30, 0.8275),
    (dominant_nuisance, 10, 10, 0.9855),
    (dominant_nuisance, 10, 30, 0.991),
]


@pytest.mark.parametrize(
    ("sample", "h", "n", "reference"),
    DIARY_COVERAGE,
    ids=[f"{sample.__name__}-H{h}-N{n}" for sample, h, n, _ in DIARY_COVERAGE],
)
def test_champion_ci_matches_the_validated_coverage(sample: GridSampler, h: int, n: int, reference: float) -> None:
    """Core's champion CI covers the best true configuration mean as often as the project's validated estimator.

    It over-covers at H=10 and under-covers the rare-outlier family at H=3 (about 0.82); those cells
    record current behaviour, so a change of method shows up as a deliberate update here.
    """
    coverage = champion_coverage(sample, h, n, np.random.default_rng(0))

    assert abs(coverage - reference) <= reference_band(reference), (coverage, reference)


CHAMPION_CELLS = [(sample, h, n) for sample, h, n, _ in DIARY_COVERAGE]


@pytest.mark.parametrize(
    ("sample", "h", "n"), CHAMPION_CELLS, ids=[f"{sample.__name__}-H{h}-N{n}" for sample, h, n in CHAMPION_CELLS]
)
def test_champion_difference_ci_is_calibrated_at_three_configs_and_conservative_at_ten(
    sample: GridSampler, h: int, n: int
) -> None:
    """The difference between two independently tuned champions is covered near nominal at H=3, conservatively at H=10.

    First measured here (seed 0, 1000 replicates), arm B's grid shifted 0.5 below arm A's:

    =================  =====  =====  ======  ======
    family             H3N10  H3N30  H10N10  H10N30
    =================  =====  =====  ======  ======
    bimodal            0.947  0.950  0.990   0.989
    long_tailed        0.957  0.952  0.996   0.992
    dominant_nuisance  0.963  0.936  0.998   0.993
    =================  =====  =====  ======  ======

    The H=10 over-coverage records current behaviour, so a change of method shows up as a deliberate update here.
    """
    coverage = champion_difference_coverage(sample, h, n, np.random.default_rng(0))

    band = 3.0 * math.sqrt(0.95 * 0.05 / REPLICATES)
    expected_low, expected_high = (0.95 - band, 0.95 + band) if h == 3 else (0.975, 1.0)
    assert expected_low <= coverage <= expected_high, coverage


@pytest.mark.parametrize(("h", "n"), [(3, 10), (3, 30), (10, 10), (10, 30)], ids=lambda value: str(value))
def test_champion_cis_stay_calibrated_when_seeds_correlate_configurations(h: int, n: int) -> None:
    """Reusing a seed across configurations and arms correlates their runs; both champion CIs still cover.

    Selection resamples each configuration's runs independently, ignoring the shared seed effect (half the
    run variance here). At H=10 both CIs stay at or above nominal. At H=3 they under-cover by a few points,
    about the percentile bootstrap's own small-sample shortfall; that range records current behaviour, so
    a change of method shows up as a deliberate update here.
    """
    rng = np.random.default_rng(0)
    means, gap = np.linspace(0.0, 1.0, h), 0.5

    single_hits = difference_hits = 0
    for _ in range(REPLICATES):
        seed_effect = math.sqrt(0.5) * rng.standard_normal(n)
        data_a = means[:, np.newaxis] + seed_effect + math.sqrt(0.5) * rng.standard_normal((h, n))
        data_b = means[:, np.newaxis] - gap + seed_effect + math.sqrt(0.5) * rng.standard_normal((h, n))
        single = hierarchical_bootstrap_ci(data_a, n, n, n_resamples=N_RESAMPLES, rng=rng)
        difference = hierarchical_bootstrap_difference_ci(data_a, data_b, n, n, n_resamples=N_RESAMPLES, rng=rng)
        single_hits += single.ci_low <= 1.0 <= single.ci_high
        difference_hits += difference.ci_low <= gap <= difference.ci_high

    nominal_floor = 0.95 - 3.0 * math.sqrt(0.95 * 0.05 / REPLICATES)
    expected_low, expected_high = (0.87, nominal_floor) if h == 3 else (nominal_floor, 1.0)
    for coverage in (single_hits / REPLICATES, difference_hits / REPLICATES):
        assert expected_low <= coverage <= expected_high, (single_hits / REPLICATES, difference_hits / REPLICATES)


def fixed_referent_difference_coverage(sample: GridSampler, h: int, n: int, rng: np.random.Generator):
    means, referent_mean = np.linspace(0.0, 1.0, h), 0.5
    hits = 0
    for _ in range(REPLICATES):
        ci = hierarchical_bootstrap_difference_ci(
            sample(rng, means, n), sample(rng, np.array([referent_mean]), n), n, n, n_resamples=N_RESAMPLES, rng=rng
        )
        hits += ci.ci_low <= 1.0 - referent_mean <= ci.ci_high
    return hits / REPLICATES


NOMINAL_BAND = 3.0 * math.sqrt(0.95 * 0.05 / REPLICATES)


@pytest.mark.parametrize(
    ("sample", "h", "n"), CHAMPION_CELLS, ids=[f"{sample.__name__}-H{h}-N{n}" for sample, h, n in CHAMPION_CELLS]
)
def test_champion_minus_untuned_referent_ci_covers_near_nominal(sample: GridSampler, h: int, n: int) -> None:
    """A tuned champion minus a single-configuration referent is covered near nominal, with a small shortfall at H=3.

    The referent has no selection stage, so its side is the plain percentile bootstrap over its runs.
    First measured here (seed 0, 1000 replicates), referent mean 0.5 against the grid's best 1.0:

    =================  =====  =====  ======  ======
    family             H3N10  H3N30  H10N10  H10N30
    =================  =====  =====  ======  ======
    bimodal            0.936  0.949  0.939   0.963
    long_tailed        0.941  0.938  0.965   0.966
    dominant_nuisance  0.928  0.912  0.986   0.968
    =================  =====  =====  ======  ======

    At H=3 the rare-outlier family under-covers by a few points; that range records current behaviour,
    so a change of method shows up as a deliberate update here.
    """
    coverage = fixed_referent_difference_coverage(sample, h, n, np.random.default_rng(0))

    expected_low = 0.90 if h == 3 else 0.95 - NOMINAL_BAND
    expected_high = 0.95 + NOMINAL_BAND if h == 3 else 1.0
    assert expected_low <= coverage <= expected_high, coverage


@pytest.mark.parametrize(("h", "n"), [(3, 10), (3, 30), (10, 10), (10, 30)], ids=lambda value: str(value))
def test_champion_minus_untuned_referent_stays_calibrated_when_seeds_correlate_the_arms(h: int, n: int) -> None:
    """Pairing by seed with a referent that shares the seed effect still covers, a few points short at H=3.

    First measured here (seed 0, 1000 replicates, half the run variance shared by seed): 0.915 at H3N10,
    0.931 at H3N30, 0.960 at H10N10, 0.950 at H10N30.
    """
    rng = np.random.default_rng(0)
    means, referent_mean = np.linspace(0.0, 1.0, h), 0.5

    hits = 0
    for _ in range(REPLICATES):
        seed_effect = math.sqrt(0.5) * rng.standard_normal(n)
        tuned = means[:, np.newaxis] + seed_effect + math.sqrt(0.5) * rng.standard_normal((h, n))
        referent = referent_mean + seed_effect[np.newaxis, :] + math.sqrt(0.5) * rng.standard_normal((1, n))
        ci = hierarchical_bootstrap_difference_ci(tuned, referent, n, n, n_resamples=N_RESAMPLES, rng=rng)
        hits += ci.ci_low <= 1.0 - referent_mean <= ci.ci_high

    coverage = hits / REPLICATES
    expected_low, expected_high = (0.87, 0.95 + NOMINAL_BAND) if h == 3 else (0.95 - NOMINAL_BAND, 1.0)
    assert expected_low <= coverage <= expected_high, coverage
