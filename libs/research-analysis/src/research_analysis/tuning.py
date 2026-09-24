from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research_analysis.bootstrap import _percentile_interval
from research_analysis.duplicates import duplicate_seed_error
from research_analysis.non_finite import require_finite_array

_DRAWS_PER_CHUNK = 1 << 20


@dataclass(frozen=True)
class HierarchicalBootstrapCI:
    """Result of hierarchical two-stage bootstrap hyperparameter evaluation."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    hyper_probs: NDArray[np.float64]
    selection_entropy: float


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _validate_hyper_matrix(arr: NDArray[np.float64], confidence: float) -> None:
    if arr.ndim != 2:
        raise ValueError(f"data must be 2D array of shape (n_hypers, n_runs), got {arr.shape}")
    n_hypers, n_runs = arr.shape
    if n_hypers < 2:
        raise ValueError(f"need at least 2 hyperparameters to tune, got {n_hypers}")
    if n_runs < 1:
        raise ValueError(f"need at least 1 run, got {n_runs}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    require_finite_array(arr, "hyperparameter matrix")


def _selection_distribution(
    data: NDArray[np.float64],
    h_samples: int,
    selection_trials: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], float]:
    """Empirical distribution of the argmax hyperparameter under a with-replacement budget of ``h_samples``.

    Drawing with replacement keeps trial-to-trial variation even at ``h_samples == n_runs``, where
    drawing the whole pool without replacement would always pick the same winner.
    """
    n_hypers, n_runs = data.shape
    counts = np.zeros(n_hypers, dtype=np.float64)
    for _ in range(selection_trials):
        draws = rng.integers(0, n_runs, size=(n_hypers, h_samples))
        counts[int(np.argmax(np.take_along_axis(data, draws, axis=1).mean(axis=1)))] += 1.0
    probs = counts / selection_trials
    selected = probs[probs > 0.0]
    return probs, float(-np.sum(selected * np.log(selected)))


def _chunk_sizes(n_resamples: int, e_samples: int) -> list[int]:
    step = max(1, _DRAWS_PER_CHUNK // e_samples)
    return [min(step, n_resamples - start) for start in range(0, n_resamples, step)]


def _mixture_means(
    data: NDArray[np.float64],
    probs: NDArray[np.float64],
    runs: NDArray[np.int64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Mean of each row of ``runs`` with a hyperparameter drawn per sample from ``probs``.

    Drawing the hyperparameter per sample rather than per replicate averages over the whole
    selection mixture, which is better calibrated when ``e_samples == h_samples == n_runs``.
    """
    hypers = rng.choice(data.shape[0], size=runs.shape, p=probs)
    return data[hypers, runs].mean(axis=1)


def hierarchical_bootstrap_ci(
    data: NDArray[np.floating],
    h_samples: int,
    e_samples: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    selection_trials: int = 200,
    rng: np.random.Generator,
) -> HierarchicalBootstrapCI:
    """Evaluate hyperparameter selection with hierarchical bootstrap resampling.

    Estimates the empirical probability distribution of the maximizing hyperparameter
    under budget ``h_samples``, and evaluates performance by sampling evaluation seeds
    from that induced distribution to prevent confidence interval collapse.
    """
    arr = np.asarray(data, dtype=np.float64)
    _validate_hyper_matrix(arr, confidence)
    _validate_positive_int(h_samples, "h_samples")
    _validate_positive_int(e_samples, "e_samples")
    _validate_positive_int(n_resamples, "n_resamples")
    _validate_positive_int(selection_trials, "selection_trials")

    probs, entropy = _selection_distribution(arr, h_samples, selection_trials, rng)
    boot_means = np.concatenate(
        [
            _mixture_means(arr, probs, rng.integers(0, arr.shape[1], size=(size, e_samples)), rng)
            for size in _chunk_sizes(n_resamples, e_samples)
        ]
    )

    ci_low, ci_high = _percentile_interval(boot_means, confidence)

    return HierarchicalBootstrapCI(
        mean=float(np.mean(boot_means)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        hyper_probs=probs,
        selection_entropy=entropy,
    )


@dataclass(frozen=True)
class HierarchicalBootstrapDifferenceCI:
    """Result of a paired hierarchical two-stage bootstrap difference evaluation (arm A minus arm B)."""

    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float
    hyper_probs_a: NDArray[np.float64]
    hyper_probs_b: NDArray[np.float64]
    selection_entropy_a: float
    selection_entropy_b: float


def hierarchical_bootstrap_difference_ci(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    h_samples: int,
    e_samples: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    selection_trials: int = 200,
    rng: np.random.Generator,
) -> HierarchicalBootstrapDifferenceCI:
    """Evaluate the difference (arm A minus arm B) between two independently-tuned grids.

    Each arm selects its champion hyperparameter independently (its own
    stage-1 draw), then evaluation samples are paired by run index across
    arms, so ``data_a`` and ``data_b`` must have the same number of columns
    and represent the same seed order, so shared seed noise cancels.
    """
    arr_a = np.asarray(data_a, dtype=np.float64)
    arr_b = np.asarray(data_b, dtype=np.float64)
    _validate_hyper_matrix(arr_a, confidence)
    _validate_hyper_matrix(arr_b, confidence)
    _validate_positive_int(h_samples, "h_samples")
    _validate_positive_int(e_samples, "e_samples")
    _validate_positive_int(n_resamples, "n_resamples")
    _validate_positive_int(selection_trials, "selection_trials")
    if arr_a.shape[1] != arr_b.shape[1]:
        raise ValueError(
            f"data_a and data_b must have the same number of paired runs, got {arr_a.shape[1]} and {arr_b.shape[1]}"
        )

    probs_a, entropy_a = _selection_distribution(arr_a, h_samples, selection_trials, rng)
    probs_b, entropy_b = _selection_distribution(arr_b, h_samples, selection_trials, rng)
    diff_chunks = []
    for size in _chunk_sizes(n_resamples, e_samples):
        runs = rng.integers(0, arr_a.shape[1], size=(size, e_samples))
        diff_chunks.append(_mixture_means(arr_a, probs_a, runs, rng) - _mixture_means(arr_b, probs_b, runs, rng))
    diffs = np.concatenate(diff_chunks)

    ci_low, ci_high = _percentile_interval(diffs, confidence)

    return HierarchicalBootstrapDifferenceCI(
        mean_diff=float(np.mean(diffs)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        hyper_probs_a=probs_a,
        hyper_probs_b=probs_b,
        selection_entropy_a=entropy_a,
        selection_entropy_b=entropy_b,
    )


def _sortable_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    """Coerce a heterogeneous hyperparameter-value tuple into a deterministic sort key."""
    return tuple(str(value) for value in key)


@dataclass(frozen=True)
class HyperMatrix:
    """Seed-paired (n_hypers, n_runs) matrix plus the seed of every run it was built from."""

    values: NDArray[np.float64]
    keys: list[tuple[Any, ...]]
    seeds: list[int]
    run_seeds: dict[int, int]

    def run_ids_outside(self, seeds: Sequence[int]) -> tuple[int, ...]:
        kept = set(seeds)
        return tuple(sorted(run_id for run_id, seed in self.run_seeds.items() if seed not in kept))


def _build_hyper_matrix(
    records: list[dict[str, Any]],
    hyperparameter_keys: Sequence[str],
) -> HyperMatrix:
    """Pivot loaded run records into a (n_hypers, n_runs) matrix, paired and ordered by seed.

    Runs are intersected down to the seeds shared by every hyperparameter
    combination so the matrix stays rectangular and each column is the same
    seed across rows -- required both for a valid hierarchical bootstrap and
    for pairing two arms' matrices in a difference evaluation.
    """
    groups: dict[tuple[Any, ...], dict[int, float]] = {}
    run_ids: dict[tuple[tuple[Any, ...], int], int] = {}
    for record in records:
        key = tuple(record["hypers"].get(k) for k in hyperparameter_keys)
        seed = record["seed"]
        if (key, seed) in run_ids:
            cell = f"configuration {dict(zip(hyperparameter_keys, key, strict=True))}"
            raise duplicate_seed_error(cell, seed, (run_ids[(key, seed)], record["run_id"]))
        run_ids[(key, seed)] = record["run_id"]
        groups.setdefault(key, {})[seed] = record["value"]

    if len(groups) < 2:
        raise ValueError(
            f"need at least 2 distinct combinations of {list(hyperparameter_keys)!r} to tune, got {len(groups)}"
        )

    keys_sorted = sorted(groups, key=_sortable_key)
    common_seeds = set.intersection(*(set(g) for g in groups.values()))
    if not common_seeds:
        raise ValueError(
            f"no seed has a completed run under every combination of {list(hyperparameter_keys)!r}"
        )
    seeds_sorted = sorted(common_seeds)

    matrix = np.array([[groups[key][s] for s in seeds_sorted] for key in keys_sorted], dtype=np.float64)
    run_seeds = {int(run_id): seed for (_key, seed), run_id in run_ids.items()}
    return HyperMatrix(matrix, keys_sorted, seeds_sorted, run_seeds)

