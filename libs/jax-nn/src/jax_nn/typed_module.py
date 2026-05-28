"""Type-safe apply() mixin for flax nn.Module subclasses."""

from __future__ import annotations

from typing import Generic, TypeVar

import jax

T_co = TypeVar("T_co", covariant=True)


class TypedApply(Generic[T_co]):
    """Generic mixin providing a typed apply() signature for flax nn.Module subclasses.

    Eliminates the ``if TYPE_CHECKING: def apply(...)`` pattern.

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
        return super().apply(variables, x, rngs=rngs)  # type: ignore[return-value]
