"""Trace-driven replay contracts for process-control benchmarks."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.sysid.trace import SignalSpec, UniformTrace

StateT = TypeVar("StateT")


def _names(specs: tuple[SignalSpec, ...]) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs)


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """Immutable selection and validation policy for one trace replay."""

    trace: UniformTrace
    start: int = 0
    stop: int | None = None
    warmup_steps: int = 0
    seed: int = 0
    required_controls: tuple[str, ...] | None = None
    required_disturbances: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        stop = self.trace.n_steps if self.stop is None else self.stop
        if self.start < 0 or stop > self.trace.n_steps or self.start >= stop:
            raise ValueError("replay row range must be non-empty and within the trace")
        horizon = stop - self.start
        if self.warmup_steps < 0 or self.warmup_steps >= horizon:
            raise ValueError("warmup_steps must be smaller than the replay horizon")

        control_names = _names(self.trace.schema.controls)
        disturbance_names = _names(self.trace.schema.disturbances)
        required_controls = (
            control_names if self.required_controls is None else self.required_controls
        )
        required_disturbances = (
            disturbance_names
            if self.required_disturbances is None
            else self.required_disturbances
        )
        if not set(required_controls).issubset(control_names):
            raise ValueError("required controls must be declared trace controls")
        if not set(required_disturbances).issubset(disturbance_names):
            raise ValueError("required disturbances must be declared trace disturbances")
        if len(set(required_controls)) != len(required_controls):
            raise ValueError("required controls must be unique")
        if len(set(required_disturbances)) != len(required_disturbances):
            raise ValueError("required disturbances must be unique")
        segment_ids = self.trace.arrays.segment_id[self.start:stop]
        if not bool(jnp.all(segment_ids == segment_ids[0])):
            raise ValueError("replay range must not cross a trace segment boundary")

        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "required_controls", tuple(required_controls))
        object.__setattr__(self, "required_disturbances", tuple(required_disturbances))

    @property
    def horizon(self) -> int:
        """Return the number of rows selected for replay."""
        if self.stop is None:
            raise RuntimeError("replay stop was not initialized")
        return self.stop - self.start

    @property
    def rows(self) -> slice:
        """Return the selected half-open trace row range."""
        return slice(self.start, self.stop)


class ReplayEnvironment(Protocol[StateT]):
    """Minimal stateful benchmark interface required by the replay runner."""

    def reset(self, key: jax.Array) -> tuple[StateT, jax.Array]: ...

    def step(
        self,
        state: StateT,
        action: jax.Array,
        key: jax.Array,
    ) -> tuple[
        StateT,
        jax.Array,
        jax.Array,
        jax.Array,
        Mapping[str, jax.Array],
    ]: ...


class ReplayEnvironmentFactory(Protocol[StateT]):
    """Construct a benchmark environment configured for one replay case."""

    def __call__(self, case: ReplayCase) -> ReplayEnvironment[StateT]: ...


@jax_dataclass
class ReplayResultArrays:
    """JAX pytree containing the time-major replay outputs."""

    time: jax.Array
    controls: jax.Array
    disturbances: jax.Array
    observations: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    diagnostics: Mapping[str, jax.Array]


@dataclass(frozen=True, slots=True)
class ReplayResult[StateT]:
    """Replay payload, source case, and terminal simulator state."""

    case: ReplayCase
    arrays: ReplayResultArrays
    final_state: StateT

    @property
    def evaluation_start(self) -> int:
        """Return the first output row outside the warmup window."""
        return self.case.warmup_steps

    @property
    def evaluation_rows(self) -> slice:
        """Return the result rows intended for evaluation."""
        return slice(self.evaluation_start, self.case.horizon)
