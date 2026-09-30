"""Named records returned by experiment planning operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from .parameter import ParameterValue


class RunBatch(NamedTuple):
    algo_version_id: int | None
    env_version_id: int | None
    ablation: str | None
    static_config: Mapping[str, ParameterValue]
    vmap_zone: Mapping[str, list[str]] | None
    run_ids: list[int]
    group_key: str


class PlannedExecution(NamedTuple):
    execution_id: int
    run_ids: list[int]
    root_path: str
    manifest_path: str
    static_config: Mapping[str, ParameterValue]
    vmap_zone: Mapping[str, list[str]] | None
