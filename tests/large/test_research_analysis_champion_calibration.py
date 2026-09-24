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
from research_analysis.tuning import hierarchical_bootstrap_ci

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
