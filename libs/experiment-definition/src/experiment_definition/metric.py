"""Metric model — whitelisted observable quantities."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class MetricType(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class MetricSpec(BaseModel):
    """Declares an observable metric that the instrumentation layer should record."""

    name: str
    type: MetricType
