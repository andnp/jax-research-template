"""Typed helpers for Chex-backed dataclasses."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast, dataclass_transform, overload

import chex

StructT = TypeVar("StructT")


@overload
@dataclass_transform(frozen_default=True)
def chex_struct(
    cls: type[StructT],
    /,
    *,
    frozen: bool = True,
    kw_only: bool = False,
    mappable_dataclass: bool = True,
) -> type[StructT]: ...


@overload
@dataclass_transform(frozen_default=True)
def chex_struct(
    *,
    frozen: bool = True,
    kw_only: bool = False,
    mappable_dataclass: bool = True,
) -> Callable[[type[StructT]], type[StructT]]: ...


def chex_struct(
    cls: type[StructT] | None = None,
    /,
    *,
    frozen: bool = True,
    kw_only: bool = False,
    mappable_dataclass: bool = True,
) -> Callable[[type[StructT]], type[StructT]] | type[StructT]:
    decorator = cast(
        Callable[[type[StructT]], type[StructT]],
        chex.dataclass(
            frozen=frozen,
            kw_only=kw_only,
            mappable_dataclass=mappable_dataclass,
        ),
    )
    if cls is None:
        return decorator
    return decorator(cls)