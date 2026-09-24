"""Fingerprint an analysis plan by what its researcher declared, leaving fields at their schema defaults out.

A field is at its default when its declared form equals that of the dataclass field's ``default``
or ``default_factory()``, both taken as JSON under the current schema. A recorded plan is read with
that schema: a field the record omits takes its default, so adding a defaulted field never changes
an existing fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

from research_analysis.plan import AnalysisPlan
from research_analysis.report_json import KIND_TAGS, JsonValue, to_jsonable

_JSON_TYPES: dict[object, tuple[type, ...]] = {
    str: (str,),
    int: (int,),
    float: (int, float),
    bool: (bool,),
    NoneType: (NoneType,),
    Path: (str,),
}


class PlanInterpretationError(ValueError):
    """A recorded plan's JSON does not fit the current plan schema."""


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is(value: JsonValue, kind: type) -> bool:
    return isinstance(value, kind) and (kind is bool or not isinstance(value, bool))


def _leaf(value: JsonValue, annotation: object) -> JsonValue:
    if get_origin(annotation) is Literal:
        accepted = any(value == option and _is(value, type(option)) for option in get_args(annotation))
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        accepted = any(value == member.value and _is(value, type(member.value)) for member in annotation)
    elif annotation in _JSON_TYPES:
        accepted = any(_is(value, kind) for kind in _JSON_TYPES[annotation])
    else:
        raise PlanInterpretationError(f"no JSON form is known for {annotation!r}")
    if not accepted:
        raise PlanInterpretationError(f"{value!r} is not a {annotation!r}")
    return value


def _members(value: JsonValue, annotation: object) -> JsonValue:
    """The one declared form ``value`` has under the union's members; members that fit it must agree."""
    forms: list[JsonValue] = []
    problems: list[str] = []
    for member in get_args(annotation):
        try:
            forms.append(_declared(value, member))
        except PlanInterpretationError as error:
            problems.append(str(error))
    if not forms:
        raise PlanInterpretationError(f"{value!r} fits no member of {annotation!r}: {'; '.join(problems)}")
    if len({_canonical(form) for form in forms}) > 1:
        raise PlanInterpretationError(f"{value!r} is ambiguous between members of {annotation!r}")
    return forms[0]


def _items(value: JsonValue, annotation: object) -> JsonValue:
    if not isinstance(value, list):
        raise PlanInterpretationError(f"{annotation!r} must be a JSON array, got {value!r}")
    arguments = get_args(annotation)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return [_declared(item, arguments[0]) for item in value]
    if len(arguments) != len(value):
        raise PlanInterpretationError(f"{annotation!r} needs {len(arguments)} items, got {value!r}")
    return [_declared(item, argument) for item, argument in zip(value, arguments, strict=True)]


def _defaults(cls: type, name: str) -> list[JsonValue]:
    """The JSON of the field's default, or nothing when the field is required."""
    item = next(item for item in fields(cls) if item.name == name)
    if item.default is not MISSING:
        return [to_jsonable(item.default)]
    if item.default_factory is not MISSING:
        return [to_jsonable(item.default_factory())]
    return []


def _dataclass(value: JsonValue, cls: type) -> JsonValue:
    """Only the fields that differ from their defaults; a kind tag is kept so union members stay distinct."""
    if not isinstance(value, dict):
        raise PlanInterpretationError(f"{cls.__name__} must be a JSON object, got {value!r}")
    data = dict(value)
    tag = KIND_TAGS.get(cls)
    if tag is not None and data.pop("kind", None) != tag:
        raise PlanInterpretationError(f"{cls.__name__} needs kind {tag!r}, got {value.get('kind')!r}")
    names = [item.name for item in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise PlanInterpretationError(f"{cls.__name__} has no field {', '.join(map(repr, unknown))}")

    hints = get_type_hints(cls)
    declared: dict[str, JsonValue] = {} if tag is None else {"kind": tag}
    for name in names:
        defaults = [_canonical(_declared(default, hints[name])) for default in _defaults(cls, name)]
        if name not in data and not defaults:
            raise PlanInterpretationError(f"{cls.__name__} is missing its required field {name!r}")
        if name not in data:
            continue
        form = _declared(data[name], hints[name])
        if _canonical(form) not in defaults:
            declared[name] = form
    return declared


def _declared(value: JsonValue, annotation: object) -> JsonValue:
    if isinstance(annotation, TypeAliasType):
        return _declared(value, annotation.__value__)
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        return _members(value, annotation)
    if origin is tuple:
        return _items(value, annotation)
    if origin is Mapping or origin is dict:
        if not isinstance(value, dict):
            raise PlanInterpretationError(f"{annotation!r} must be a JSON object, got {value!r}")
        item_annotation = get_args(annotation)[1]
        return {key: _declared(item, item_annotation) for key, item in value.items()}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass(value, annotation)
    return _leaf(value, annotation)


def recorded_plan_fingerprint(plan_json: JsonValue) -> str:
    """Fingerprint of a recorded plan's JSON read with the current schema; raises ``PlanInterpretationError``."""
    canonical = _canonical(_dataclass(plan_json, AnalysisPlan))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def plan_fingerprint(plan: AnalysisPlan) -> str:
    """Stable content hash of the plan's declared fields, identical across processes for identical declarations."""
    return recorded_plan_fingerprint(to_jsonable(plan))
