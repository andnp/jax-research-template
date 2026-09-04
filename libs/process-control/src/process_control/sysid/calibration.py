"""Generic parameter-space contracts for simulator calibration."""

from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class CalibrationParameter:
    """Metadata for one bounded calibration parameter."""

    name: str
    initial: float
    lower: float
    upper: float
    scale: float
    group: str

    def __post_init__(self) -> None:
        if not self.name or not self.group:
            raise ValueError("parameter name and group must not be empty")
        values = (self.initial, self.lower, self.upper, self.scale)
        if not all(isfinite(value) for value in values):
            raise ValueError("parameter values must be finite")
        if self.lower > self.upper:
            raise ValueError("parameter lower bound must not exceed upper bound")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("parameter initial value must be within its bounds")
        if self.scale <= 0.0:
            raise ValueError("parameter scale must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationParameterSpace:
    """Ordered bounded parameter metadata with JAX numeric views."""

    parameters: tuple[CalibrationParameter, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("at least one calibration parameter is required")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("calibration parameter names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return parameter names in vector order."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def groups(self) -> tuple[str, ...]:
        """Return distinct parameter groups in declaration order."""
        return tuple(dict.fromkeys(parameter.group for parameter in self.parameters))

    @property
    def initial_vector(self) -> jax.Array:
        """Return the initial parameter vector."""
        return jnp.asarray(
            [parameter.initial for parameter in self.parameters],
            dtype=jnp.float32,
        )

    @property
    def lower_bounds(self) -> jax.Array:
        """Return lower bounds in vector order."""
        return jnp.asarray(
            [parameter.lower for parameter in self.parameters],
            dtype=jnp.float32,
        )

    @property
    def upper_bounds(self) -> jax.Array:
        """Return upper bounds in vector order."""
        return jnp.asarray(
            [parameter.upper for parameter in self.parameters],
            dtype=jnp.float32,
        )

    @property
    def scales(self) -> jax.Array:
        """Return normalization scales in vector order."""
        return jnp.asarray(
            [parameter.scale for parameter in self.parameters],
            dtype=jnp.float32,
        )

    def indices_for_groups(self, groups: tuple[str, ...]) -> jax.Array:
        """Return vector indices for selected groups in declaration order."""
        selected = set(groups)
        unknown = selected.difference(self.groups)
        if unknown:
            raise ValueError(f"unknown calibration parameter groups: {sorted(unknown)}")
        return jnp.asarray(
            [
                index
                for index, parameter in enumerate(self.parameters)
                if parameter.group in selected
            ],
            dtype=jnp.int32,
        )
