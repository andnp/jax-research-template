"""Reference-based benchmark aggregates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReferenceRange:
    """Fixed score anchors for one benchmark metric."""

    low: float
    high: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high) or self.low >= self.high:
            raise ValueError("reference range must be finite with low < high")


def normalize_scores(values: NDArray[np.floating], reference: ReferenceRange) -> NDArray[np.float64]:
    """Normalize scores against fixed anchors without pooling comparison arms."""
    scores = np.asarray(values, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    normalized = (scores - reference.low) / (reference.high - reference.low)
    if not reference.higher_is_better:
        normalized = 1.0 - normalized
    return np.ascontiguousarray(normalized)


def performance_profile(
    scores: NDArray[np.floating],
    thresholds: Sequence[float],
    labels: Sequence[str],
) -> dict[str, dict[float, float]]:
    """Return the fraction of units above each normalized-score threshold."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels) or matrix.shape[0] < 1:
        raise ValueError("scores must be a non-empty 2-D matrix matching labels")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must contain only finite values")
    threshold_values = [float(value) for value in thresholds]
    if any(not np.isfinite(value) for value in threshold_values):
        raise ValueError("thresholds must contain only finite values")
    return {
        label: {
            threshold: float(np.mean(matrix[:, index] >= threshold))
            for threshold in threshold_values
        }
        for index, label in enumerate(labels)
    }


def probability_of_improvement(values_a: NDArray[np.floating], values_b: NDArray[np.floating]) -> float:
    """Return paired probability that A improves over B, counting ties halfway."""
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if len(a) != len(b) or len(a) < 1:
        raise ValueError("paired improvement samples must have equal non-empty lengths")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("improvement samples must contain only finite values")
    return float(np.mean((a > b) + 0.5 * (a == b)))


def optimality_gap(scores: NDArray[np.floating], reference: float = 1.0) -> float:
    """Return the mean gap from a fixed normalized reference score."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if len(values) < 1 or not np.isfinite(values).all() or not np.isfinite(reference):
        raise ValueError("optimality-gap inputs must be finite and non-empty")
    return float(np.mean(reference - values))


def interquartile_mean(scores: NDArray[np.floating], trim_fraction: float = 0.25) -> float:
    """Return IQM as a supplementary center summary."""
    values = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("IQM requires at least two finite scores")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    lower = int(np.floor(len(values) * trim_fraction))
    upper = len(values) - lower
    if lower >= upper:
        raise ValueError("trim_fraction leaves no scores for IQM")
    return float(np.mean(values[lower:upper]))
