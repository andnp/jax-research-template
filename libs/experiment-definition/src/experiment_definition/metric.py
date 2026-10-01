"""Metric model — the emitted quantities an experiment records, and how often."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, StrictInt

from .parameter import ParameterValue


class MetricType(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class MetricSpec(BaseModel):
    """One recorded metric: smoothed by an ``ema`` (if any), then one value kept in every ``every`` steps.

    ``conditions`` limit the metric to runs whose parameters match all of them.
    """

    name: str
    type: MetricType
    every: StrictInt = Field(default=1, ge=1)
    ema: float | None = Field(default=None, ge=0, lt=1)
    conditions: dict[str, ParameterValue] = {}
