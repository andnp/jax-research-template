"""Config fields that may vary between the runs of one batch."""

import dataclasses
from collections.abc import Iterator, Mapping
from types import UnionType
from typing import TYPE_CHECKING, Literal, Union, get_args, get_origin

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_TRACED = "research_runner.traced"


def traced[T](default: T) -> T:
    """Mark a config field as vmappable: a run reads it only in JAX arithmetic."""
    return dataclasses.field(default=default, metadata={_TRACED: True})


def _walk(config: object, prefix: str = "") -> Iterator[tuple[str, dataclasses.Field]]:
    """Yield each traced field with its dotted path, by runtime value. A traced field is never entered."""
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        return
    for field in dataclasses.fields(config):
        path = f"{prefix}{field.name}"
        if field.metadata.get(_TRACED):
            yield path, field
        else:
            yield from _walk(getattr(config, field.name), f"{path}.")


def _traced_paths(config: object) -> tuple[str, ...]:
    return tuple(path for path, _ in _walk(config))


def _get_path(config: object, path: str) -> object:
    for part in path.split("."):
        config = getattr(config, part)
    return config


def _set_path[C: DataclassInstance](config: C, path: str, value: object) -> C:
    head, _, rest = path.partition(".")
    names = [field.name for field in dataclasses.fields(config)] if dataclasses.is_dataclass(config) else []
    if head not in names:
        raise ValueError(f"{type(config).__name__} has no field {head}; fields are {', '.join(names) or 'none'}")
    if rest:
        value = _set_path(getattr(config, head), rest, value)
    return dataclasses.replace(config, **{head: value})


def _split_traced[C: DataclassInstance](config: C) -> tuple[C, dict[str, object]]:
    """Reset every traced field to its default and return the values it held, so equal statics can share a batch."""
    static = config
    values: dict[str, object] = {}
    for path, field in _walk(config):
        values[path] = _get_path(config, path)
        static = _set_path(static, path, field.default)
    return static, values


def _merge_traced[C: DataclassInstance](static: C, values: Mapping[str, object]) -> C:
    """Set traced values, which may be tracers, back onto a static config."""
    config = static
    for path, value in values.items():
        config = _set_path(config, path, value)
    return config


def _check_value(annotation: object, path: str, value: object) -> object:
    """Return the value as its field's type, or raise if it is not one. Annotations not known here pass through."""
    if get_origin(annotation) in (Union, UnionType):
        members = [member for member in get_args(annotation) if member is not type(None)]
        if value is None and len(members) < len(get_args(annotation)):
            return value
        for member in members:
            try:
                return _check_value(member, path, value)
            except ValueError:
                continue
        raise ValueError(f"{path} expects {annotation}, got {value!r}")
    if get_origin(annotation) is Literal:
        options = get_args(annotation)
        if any(type(option) is type(value) and option == value for option in options):
            return value
        raise ValueError(f"{path} expects one of {options}, got {value!r}")
    if not isinstance(annotation, type):
        return value
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    if annotation in (bool, str) and isinstance(value, annotation):
        return value
    if annotation is int and is_number and float(value).is_integer():
        return int(value)
    if annotation is float and is_number:
        return float(value)
    if annotation in (bool, int, float, str):
        raise ValueError(f"{path} expects {annotation.__name__}, got {value!r}")
    return value
