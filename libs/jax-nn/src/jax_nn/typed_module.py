"""Type-safe apply() mixin for flax nn.Module subclasses."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar, cast

import jax

T_co = TypeVar("T_co", covariant=True)


class _SupportsApply(Protocol[T_co]):
    """The apply() the sibling nn.Module base contributes at runtime via the MRO."""

    def apply(self, variables: object, x: jax.Array, *, rngs: object | None = None) -> T_co: ...


class TypedApply(Generic[T_co]):
    """Generic mixin providing a typed apply() signature for flax nn.Module subclasses.

    Usage (always put TypedApply before nn.Module in the bases)::

        class MyNet(TypedApply[jax.Array], nn.Module):
            ...

    At runtime, the MRO calls this apply() which delegates to nn.Module.apply().
    For type checkers, the return type T_co is inferred from the generic parameter.
    """

    def apply(
        self,
        variables: object,
        x: jax.Array,
        *,
        rngs: object | None = None,
    ) -> T_co:
        return cast(_SupportsApply[T_co], super()).apply(variables, x, rngs=rngs)
