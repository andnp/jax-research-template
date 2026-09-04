"""Multiple-comparison adjustments."""

from __future__ import annotations

import math
from collections.abc import Sequence


def holm_step_down(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in input order."""
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be finite numbers in [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    count = len(p_values)
    for rank, (index, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * p_value))
        adjusted[index] = running_max
    return adjusted
