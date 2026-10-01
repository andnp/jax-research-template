"""Config fields that may vary between the runs of one batch."""

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from types import UnionType
from typing import TYPE_CHECKING, Literal, Union, get_args, get_origin, get_type_hints

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


def _require_field(config: object, name: str) -> None:
    names = [field.name for field in dataclasses.fields(config)] if dataclasses.is_dataclass(config) else []
    if name not in names:
        raise ValueError(f"{type(config).__name__} has no field {name}; fields are {', '.join(names) or 'none'}")


def _set_path[C: DataclassInstance](config: C, path: str, value: object) -> C:
    head, _, rest = path.partition(".")
    _require_field(config, head)
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
    number = value if isinstance(value, int | float) and not isinstance(value, bool) else None
    if annotation in (bool, str) and isinstance(value, annotation):
        return value
    if annotation is int and number is not None and float(number).is_integer():
        return int(number)
    if annotation is float and number is not None:
        return float(number)
    if annotation in (bool, int, float, str):
        raise ValueError(f"{path} expects {annotation.__name__}, got {value!r}")
    return value


def _apply_choices[C: DataclassInstance](config: C, choices: Mapping[str, Mapping[str, object]], params: Mapping[str, object]) -> C:
    """Swap in the named object for each choice a run's parameters select. Unselected choices keep the base object."""
    for path, options in choices.items():
        if path not in params:
            continue
        name = params[path]
        if not isinstance(name, str) or name not in options:
            raise ValueError(f"{name!r} is not a choice for {path}; choices are {', '.join(options)}")
        config = _set_path(config, path, options[name])
    return config


def _field_annotation(config: object, path: str) -> object:
    *parents, leaf = path.split(".")
    owner = config
    for name in parents:
        _require_field(owner, name)
        owner = getattr(owner, name)
    _require_field(owner, leaf)
    return get_type_hints(type(owner))[leaf]


def _build_config[C: DataclassInstance](base: C, choices: Mapping[str, Mapping[str, object]], params: Mapping[str, object]) -> C:
    """Build one run's config: choices first, then each dotted path, type-checked. `seed` is not a config field."""
    config = _apply_choices(base, choices, params)
    for path, value in params.items():
        if path == "seed" or path in choices:
            continue
        config = _set_path(config, path, _check_value(_field_annotation(config, path), path, value))
    return config


def _static_names(config: object, names: Iterable[str]) -> frozenset[str]:
    """The parameter names that decide which batch a run joins: those not setting a traced field, and not `seed`."""
    traced = set(_traced_paths(config))
    return frozenset(name for name in names if name != "seed" and name not in traced)
