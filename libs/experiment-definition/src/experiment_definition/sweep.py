"""Standalone conditional parameter sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

SweepValue = int | float | str | bool | None


@dataclass(frozen=True)
class _Values:
    items: tuple[SweepValue, ...]


def values(*items: SweepValue) -> _Values:
    return _Values(items)


@dataclass(frozen=True)
class _Constraint:
    condition_name: str
    condition_value: SweepValue
    requirements: tuple[tuple[str, SweepValue], ...] = ()

    def require(self, name: str, value: SweepValue) -> _Constraint:
        requirements = dict(self.requirements)
        requirements[name] = value
        return _Constraint(self.condition_name, self.condition_value, tuple(requirements.items()))

    def matches(self, configuration: Mapping[str, SweepValue]) -> bool:
        if configuration[self.condition_name] != self.condition_value:
            return True
        return all(configuration[name] == value for name, value in self.requirements)


def when(name: str, value: SweepValue) -> _Constraint:
    return _Constraint(name, value)


@dataclass(frozen=True)
class Sweep:
    axes: Mapping[str, _Values]
    constraints: Sequence[_Constraint] = ()

    def __post_init__(self) -> None:
        axes = dict(self.axes)
        if any(not name for name in axes):
            raise ValueError("Sweep axis names must be non-empty")
        if any(not isinstance(axis_values, _Values) for axis_values in axes.values()):
            raise TypeError("Sweep axes must use values(...)")

        constraints = tuple(self.constraints)
        axis_names = set(axes)
        for constraint in constraints:
            if not isinstance(constraint, _Constraint):
                raise TypeError("Sweep constraints must use when(...).require(...)")
            referenced_names = {constraint.condition_name} | {name for name, _ in constraint.requirements}
            unknown_names = referenced_names - axis_names
            if unknown_names:
                names = ", ".join(sorted(unknown_names))
                raise ValueError(f"Sweep constraint references unknown axis: {names}")

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "constraints", constraints)

    def resolve(self) -> list[dict[str, SweepValue]]:
        axis_items = tuple(self.axes.items())
        axis_names = tuple(name for name, _ in axis_items)
        axis_values = tuple(axis.items for _, axis in axis_items)
        configurations = (dict(zip(axis_names, combination, strict=True)) for combination in product(*axis_values))
        return [configuration for configuration in configurations if all(constraint.matches(configuration) for constraint in self.constraints)]
