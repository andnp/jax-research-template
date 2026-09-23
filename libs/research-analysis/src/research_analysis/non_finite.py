"""Fail loudly on diverged (NaN or infinite) run values unless a policy is declared explicitly."""

from __future__ import annotations

import polars as pl

_IDENTITY_COLUMNS = ("condition_name", "env_name", "run_id", "seed")
_LISTED_RUNS = 10


class NonFiniteValuesError(ValueError):
    """Raised when a statistic would consume NaN or infinite run values."""

    def __init__(self, context: str, offenders: pl.DataFrame) -> None:
        self.offenders = offenders
        listed = offenders.head(_LISTED_RUNS).to_dicts()
        more = f" and {offenders.height - _LISTED_RUNS} more" if offenders.height > _LISTED_RUNS else ""
        super().__init__(
            f"{context} has non-finite values for {listed}{more}; diverged runs are outcomes, not missing data. "
            "Declare a worst-case value with research_analysis.non_finite.substitute_non_finite or analyse "
            "divergence separately."
        )


def require_finite(frame: pl.DataFrame, context: str) -> None:
    offenders = frame.filter(~pl.col("value").is_finite())
    if offenders.is_empty():
        return
    columns = [column for column in _IDENTITY_COLUMNS if column in frame.columns]
    raise NonFiniteValuesError(context, offenders.select(columns).unique(maintain_order=True))
