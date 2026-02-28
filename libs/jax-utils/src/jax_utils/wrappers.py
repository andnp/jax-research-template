"""Typed wrappers for jax.jit and jax.vmap that preserve function signatures."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import jax


@overload
def typed_jit[**P, R](fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def typed_jit[**P, R](fn: Callable[P, R], *, donate_argnums: int | tuple[int, ...] = ...) -> Callable[P, R]: ...


def typed_jit[**P, R](
    fn: Callable[P, R],
    *,
    donate_argnums: int | tuple[int, ...] = (),
) -> Callable[P, R]:
    """JIT-compile *fn* while preserving its type signature.

    This is a thin wrapper around ``jax.jit`` that keeps the input/output
    types visible to type checkers, making call sites type-safe.

    Args:
        fn: The function to compile.
        donate_argnums: Argument indices whose buffers may be donated.

    Returns:
        A JIT-compiled callable with the same signature as *fn*.
    """
    return jax.jit(fn, donate_argnums=donate_argnums)  # type: ignore[return-value]


def typed_vmap[**P, R](
    fn: Callable[P, R],
    *,
    in_axes: int | tuple[object, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
    axis_name: str | None = None,
) -> Callable[P, R]:
    """Vectorise *fn* while preserving its type signature.

    This is a thin wrapper around ``jax.vmap`` that keeps the input/output
    types visible to type checkers.

    Args:
        fn: The function to vectorise.
        in_axes: Specification of which input axes to map over.
        out_axes: Specification of where the mapped axis appears in outputs.
        axis_name: Optional name for the mapped axis (for collective ops).

    Returns:
        A vectorised callable with the same signature as *fn*.
    """
    return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name)  # type: ignore[return-value]
