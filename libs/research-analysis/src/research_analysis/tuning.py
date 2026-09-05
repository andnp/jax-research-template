from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HierarchicalBootstrapCI:
    """Result of hierarchical two-stage bootstrap hyperparameter evaluation."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    hyper_probs: NDArray[np.float64]
    selection_entropy: float
