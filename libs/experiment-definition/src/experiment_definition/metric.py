"""Metric model — whitelisted observable quantities."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class MetricFrequency(StrEnum):
    PER_EPISODE = "per_episode"
    PER_UPDATE = "per_update"
    EVAL_ONLY = "eval_only"


class MetricType(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class MetricSpec(BaseModel):
    """Declares an observable metric that the instrumentation layer should record.

    Args:
        name: Identifier used by the collector (e.g. "reward").
        type: Data type of the scalar.
        frequency: Optional legacy collection cadence.
    """

    name: str
    type: MetricType
    # Remove when all core and project callers stop passing frequency.
    frequency: MetricFrequency | None = None
