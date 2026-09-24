"""Serialize report documents and analysis plans to strict JSON."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from research_analysis.dataset.types import CellAccounting
from research_analysis.diagnostics.bakeoff import BakeoffWarning
from research_analysis.performance import FinalValue, ReturnRate, TailMean
from research_analysis.plan import (
    AnalysisPlan,
    BakeoffAnalysis,
    BestHypersAnalysis,
    HypersDifferenceAnalysis,
    SliceHypersAnalysis,
)
from research_analysis.report import (
    BakeoffResult,
    BestHypersResult,
    HypersDifferenceResult,
    ReportDocument,
    SliceHypersResult,
)

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

# Members of a union are tagged so a reader can tell, e.g., FinalValue() from ReturnRate().
_KINDS: dict[type, str] = {
    FinalValue: "final_value",
    TailMean: "tail_mean",
    ReturnRate: "return_rate",
    BakeoffAnalysis: "bakeoff",
    BestHypersAnalysis: "best_hypers",
    HypersDifferenceAnalysis: "hypers_difference",
    SliceHypersAnalysis: "slice_hypers",
    BakeoffResult: "bakeoff",
    BestHypersResult: "best_hypers",
    HypersDifferenceResult: "hypers_difference",
    SliceHypersResult: "slice_hypers",
}

_ANALYSES = (BakeoffAnalysis, BestHypersAnalysis, HypersDifferenceAnalysis, SliceHypersAnalysis)


def _tagged(value: object, data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if isinstance(value, CellAccounting | BakeoffWarning):
        data["cell"] = {axis: to_jsonable(axis_value) for axis, axis_value in value.cell}
    kind = _KINDS.get(type(value))
    return data if kind is None else {"kind": kind, **data}


def to_jsonable(value: object) -> JsonValue:
    """Convert report values to JSON types; mappings must have string keys."""
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Enum():
            return to_jsonable(value.value)
        case Path():
            return str(value)
        case np.generic():
            return to_jsonable(value.item())
        case Mapping():
            if not all(isinstance(key, str) for key in value):
                raise TypeError(f"report mappings need string keys, got {list(value)!r}")
            return {str(key): to_jsonable(item) for key, item in value.items()}
        case tuple() | list():
            return [to_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        attributes = vars(value)
        data = {item.name: to_jsonable(attributes[item.name]) for item in fields(value)}
        # An analysis inheriting the plan's measure omits the field, so plans that never override keep their fingerprint.
        if isinstance(value, _ANALYSES) and value.measure is None:
            del data["measure"]
        return _tagged(value, data)
    raise TypeError(f"cannot serialize {type(value).__name__} in a report")


def report_to_json(document: ReportDocument) -> str:
    """Strict JSON: a NaN or infinity anywhere in the report raises instead of writing invalid JSON."""
    return json.dumps(to_jsonable(document), indent=2, allow_nan=False) + "\n"


def plan_fingerprint(plan: AnalysisPlan) -> str:
    """Stable content hash of a plan, identical across processes for identical declarations."""
    canonical = json.dumps(to_jsonable(plan), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
