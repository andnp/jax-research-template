"""Validated records returned by the experiment database."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, field_validator

from .parameter import ParameterValue


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ComponentRow(_Record):
    id: int
    name: str
    type: str


class ComponentVersionRow(_Record):
    id: int
    component_id: int
    version_number: int
    created_at: str
    code_snapshot_hash: str
    notes: str | None


class ComponentVersionProvenanceRow(_Record):
    component_id: int
    component_name: str
    component_type: str
    version_id: int
    version_number: int
    code_snapshot_hash: str


class HyperparamConfigRow(_Record):
    id: int
    hash: str
    hyperparameters: Mapping[str, ParameterValue]


class ExperimentRow(_Record):
    id: int
    name: str
    description: str | None
    created_at: str


class RunRow(_Record):
    id: int
    experiment_id: int
    algo_version_id: int | None
    env_version_id: int | None
    hyper_id: int
    seed: int
    ablation: str | None
    vmap_zone: Mapping[str, list[str]] | None

    @field_validator("vmap_zone", mode="before")
    @classmethod
    def _decode_vmap_zone(cls, value: object):
        return json.loads(value) if isinstance(value, str) else value


def _run_matches(hyperparams: Mapping[str, ParameterValue], run: RunRow, only: Mapping[str, str]) -> bool:
    fields: dict[str, object] = {**hyperparams, "seed": run.seed, "ablation": run.ablation}
    return all(
        key in fields and str(fields[key]) in set(value.split(","))
        for key, value in only.items()
    )


class ExecutionRow(_Record):
    id: int
    status: str
    hostname: str | None
    start_time: str | None
    end_time: str | None
    git_commit: str | None
    git_diff_blob: str | None
    jax_config_json: str | None


class ExecutionArtifactRow(_Record):
    execution_id: int
    root_path: str
    manifest_path: str | None
    metadata_json: str | None
    created_at: str


def _record_from_row[T: _Record](model: type[T], row: sqlite3.Row) -> T:
    values = {field: row[field] for field in model.model_fields}
    return model.model_validate(values)
