"""Typed, array-oriented traces for process-control system identification."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


class SignalRole(StrEnum):
    """Semantic role of a signal in a closed-loop process."""

    CONTROL = "control"
    DISTURBANCE = "disturbance"
    MEASUREMENT = "measurement"
    ACTUATOR = "actuator"


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Immutable metadata describing one signal."""

    name: str
    role: SignalRole
    unit: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name must not be empty")
        if not self.unit:
            raise ValueError("signal unit must not be empty")


@dataclass(frozen=True, slots=True)
class TraceSchema:
    """Signal metadata grouped by closed-loop process role."""

    controls: tuple[SignalSpec, ...] = ()
    disturbances: tuple[SignalSpec, ...] = ()
    measurements: tuple[SignalSpec, ...] = ()
    actuators: tuple[SignalSpec, ...] = ()

    def __post_init__(self) -> None:
        specs = self.controls + self.disturbances + self.measurements + self.actuators
        names = tuple(spec.name for spec in specs)
        if len(set(names)) != len(names):
            raise ValueError("signal names must be unique")
        groups = (
            (self.controls, SignalRole.CONTROL),
            (self.disturbances, SignalRole.DISTURBANCE),
            (self.measurements, SignalRole.MEASUREMENT),
            (self.actuators, SignalRole.ACTUATOR),
        )
        for group, role in groups:
            if any(spec.role is not role for spec in group):
                raise ValueError(f"signal metadata role must be {role.value!r}")

    @property
    def width(self) -> int:
        """Return the total number of signals in the schema."""
        return sum(
            len(group)
            for group in (
                self.controls,
                self.disturbances,
                self.measurements,
                self.actuators,
            )
        )


@dataclass(frozen=True, slots=True)
class SignalSeries:
    """One signal sampled at its native, potentially irregular timestamps."""

    time: jax.Array
    values: jax.Array
    spec: SignalSpec
    valid: jax.Array | None = None

    def __post_init__(self) -> None:
        time = jnp.asarray(self.time)
        values = jnp.asarray(self.values)
        if time.ndim != 1 or values.ndim != 1:
            raise ValueError("signal time and values must be one-dimensional")
        if time.shape != values.shape:
            raise ValueError("signal time and values must have equal lengths")
        if time.shape[0] == 0:
            raise ValueError("signal series must not be empty")
        if not bool(jnp.all(jnp.isfinite(time))):
            raise ValueError("signal timestamps must be finite")
        if not bool(jnp.all(jnp.diff(time) > 0.0)):
            raise ValueError("signal timestamps must be strictly increasing")
        valid = jnp.isfinite(values) if self.valid is None else jnp.asarray(self.valid)
        if valid.shape != values.shape or valid.dtype != jnp.bool_:
            raise ValueError("signal validity must be a boolean vector matching values")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)

    @property
    def validity(self) -> jax.Array:
        """Return the validated boolean mask for this signal."""
        if self.valid is None:
            raise RuntimeError("signal validity was not initialized")
        return self.valid


@jax_dataclass
class UniformTraceArrays:
    """JAX pytree containing only the numeric payload of a uniform trace."""

    time: jax.Array
    controls: jax.Array
    disturbances: jax.Array
    measurements: jax.Array
    actuators: jax.Array
    valid: jax.Array
    source_count: jax.Array
    segment_id: jax.Array


@dataclass(frozen=True, slots=True)
class UniformTrace:
    """Validated SoA trace on one regular, fixed-step time grid."""

    arrays: UniformTraceArrays
    schema: TraceSchema
    sample_period: float

    def __post_init__(self) -> None:
        arrays = self.arrays
        if not jnp.isfinite(self.sample_period) or self.sample_period <= 0.0:
            raise ValueError("sample_period must be finite and positive")
        if arrays.time.ndim != 1 or arrays.time.shape[0] == 0:
            raise ValueError("uniform trace time must be a non-empty vector")
        if not bool(jnp.all(jnp.isfinite(arrays.time))):
            raise ValueError("uniform trace timestamps must be finite")
        if not bool(
            jnp.all(
                jnp.isclose(
                    jnp.diff(arrays.time),
                    self.sample_period,
                    rtol=0.0,
                    atol=self.sample_period * 1e-9,
                )
            )
        ):
            raise ValueError("uniform trace timestamps must use sample_period spacing")
        steps = arrays.time.shape[0]
        groups = (
            (arrays.controls, len(self.schema.controls), "controls"),
            (arrays.disturbances, len(self.schema.disturbances), "disturbances"),
            (arrays.measurements, len(self.schema.measurements), "measurements"),
            (arrays.actuators, len(self.schema.actuators), "actuators"),
        )
        for values, width, name in groups:
            if values.shape != (steps, width):
                raise ValueError(f"{name} must have shape ({steps}, {width})")
        if arrays.valid.shape != (steps, self.schema.width):
            raise ValueError("valid must have one column per schema signal")
        if arrays.valid.dtype != jnp.bool_:
            raise ValueError("valid must be a boolean matrix")
        if arrays.source_count.shape != (steps, self.schema.width):
            raise ValueError("source_count must have one column per schema signal")
        if arrays.source_count.dtype.kind not in "iu" or bool(jnp.any(arrays.source_count < 0)):
            raise ValueError("source_count must be a non-negative integer matrix")
        if arrays.segment_id.shape != (steps,) or arrays.segment_id.dtype.kind not in "iu":
            raise ValueError("segment_id must be an integer vector matching time")

    @property
    def time(self) -> jax.Array:
        """Return the uniform time grid."""
        return self.arrays.time

    @property
    def n_steps(self) -> int:
        """Return the number of rows in the trace."""
        return self.arrays.time.shape[0]

    @property
    def jax_payload(self) -> UniformTraceArrays:
        """Return the metadata-free pytree for JAX functions."""
        return self.arrays
