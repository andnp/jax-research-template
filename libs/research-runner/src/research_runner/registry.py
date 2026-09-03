"""Registration of experiment spec factories for CLI discovery."""

from __future__ import annotations

from collections.abc import Callable

from .types import ExperimentSpec

SPECS_ATTRIBUTE = "__research_specs__"

SpecFactory = Callable[[], ExperimentSpec]


def spec(factory: SpecFactory) -> SpecFactory:
    """Register an experiment spec factory in its defining module."""
    registry: dict[str, SpecFactory] = factory.__globals__.setdefault(SPECS_ATTRIBUTE, {})
    registry[factory.__name__] = factory
    return factory


def registered_specs(module: object) -> dict[str, SpecFactory]:
    """Return the spec factories registered in a loaded spec module."""
    return dict(getattr(module, SPECS_ATTRIBUTE, {}))
