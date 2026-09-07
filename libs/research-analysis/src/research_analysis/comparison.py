"""Validated configuration for statistical comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type Estimand = Literal["mean", "median"]
type Pairing = Literal["auto", "paired", "independent"]
type Missingness = Literal["error", "complete_case", "blockwise"]


@dataclass(frozen=True)
class BakeoffSpec:
    """Declare the estimand, uncertainty, and missingness policy for a bakeoff."""

    estimand: Estimand = "mean"
    alpha: float = 0.05
    n_resamples: int = 10_000
    min_samples: int = 3
    missingness: Missingness = "error"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.estimand not in ("mean", "median"):
            raise ValueError(f"estimand must be 'mean' or 'median', got {self.estimand!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.n_resamples < 1:
            raise ValueError(f"n_resamples must be positive, got {self.n_resamples}")
        if self.min_samples < 3:
            raise ValueError(f"min_samples must be at least 3, got {self.min_samples}")
        if self.missingness not in ("error", "complete_case", "blockwise"):
            raise ValueError(f"unknown missingness policy {self.missingness!r}")

    @property
    def confidence_level(self) -> float:
        """Return the confidence level implied by ``alpha``."""
        return 1.0 - self.alpha
