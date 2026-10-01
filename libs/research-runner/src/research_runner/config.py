"""Config fields that may vary between the runs of one batch."""

import dataclasses
from collections.abc import Iterator

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
