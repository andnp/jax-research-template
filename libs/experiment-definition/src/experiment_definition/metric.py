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
        frequency: How often the metric is expected to be written.
    """

    name: str
    type: MetricType
    frequency: MetricFrequency
