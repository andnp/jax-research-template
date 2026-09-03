"""``@jax_dataclass`` — single-decorator wrapper for JAX pytree dataclasses.

Replaces the two-step song-and-dance::

    @dataclass(frozen=True)
    class MyState:
        x: jax.Array
        y: jax.Array

    jax.tree_util.register_dataclass(
        MyState,
        data_fields=["x", "y"],
        meta_fields=[],
    )

With a single decorator::

    @jax_dataclass
    class MyState:
        x: jax.Array
        y: jax.Array

The decorator registers *every* dataclass field as a JAX pytree data field
(``meta_fields`` always empty), which is the universal pattern across this
codebase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import dataclass_transform, overload

import jax


@overload
@dataclass_transform(frozen_default=True)
def jax_dataclass[T](cls: type[T], /) -> type[T]: ...


@overload
@dataclass_transform(frozen_default=True)
def jax_dataclass[T](*, frozen: bool = True) -> Callable[[type[T]], type[T]]: ...


def jax_dataclass[T](
    cls: type[T] | None = None,
    /,
    *,
    frozen: bool = True,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Decorator combining ``@dataclass(frozen=...)`` with automatic pytree registration.

    All dataclass fields are registered as ``data_fields`` (the universal pattern
    in this codebase — ``meta_fields`` is always empty).
    """

    def _wrap(c: type[T]) -> type[T]:
        # Read annotation keys before @dataclass is applied (they are the
        # same as the eventual __dataclass_fields__ keys).
        fields = list(c.__annotations__)
        dc = dataclass(frozen=frozen)(c)
        jax.tree_util.register_dataclass(dc, data_fields=fields, meta_fields=[])
        return dc

    if cls is None:
        return _wrap
    return _wrap(cls)
