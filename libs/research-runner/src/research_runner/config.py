"""Config fields that may vary between the runs of one batch."""

import dataclasses
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

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
