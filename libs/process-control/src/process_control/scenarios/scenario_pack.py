"""Composable, versioned scenario packs for process-control episodes."""

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

import jax
import jax.numpy as jnp
from jax import Array

from process_control.environment_specs import ScenarioSpec, SignalSpec


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty and have no surrounding whitespace")


def _require_unique(values: tuple[str, ...], kind: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {kind}: {', '.join(duplicates)}")


def _require_finite(value: float, field: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")


class RandomStream(StrEnum):
    """Independent sources of episode randomness."""

    PROCESS = "process"
    SENSOR = "sensor"
    FAULT = "fault"
    SCENARIO = "scenario"
    CONTROLLER = "controller"


_STREAM_IDS = {
    RandomStream.PROCESS: 0,
    RandomStream.SENSOR: 1,
    RandomStream.FAULT: 2,
    RandomStream.SCENARIO: 3,
    RandomStream.CONTROLLER: 4,
}


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Named, independently addressable PRNG roots for one episode."""

    root_seed: int
    process: Array
    sensor: Array
    fault: Array
    scenario: Array
    controller: Array

    @classmethod
    def from_seed(cls, seed: int) -> "SeedBundle":
        """Derive stable stream roots from a single episode seed."""
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("scenario seed must be an integer")
        seed = int(seed)
        if seed < 0:
            raise ValueError("scenario seed must be non-negative")
        root = jax.random.key(seed)
        keys = tuple(jax.random.fold_in(root, _STREAM_IDS[stream]) for stream in RandomStream)
        return cls(seed, *keys)

    def key(self, stream: RandomStream, index: int = 0) -> Array:
        """Return a deterministic subkey without advancing any other stream."""
        if index < 0:
            raise ValueError("random stream index must be non-negative")
        return jax.random.fold_in(getattr(self, stream.value), index)


@dataclass(frozen=True, slots=True)
class InitialValue:
    """Named initial-state value with units."""

    name: str
    units: str
    value: float

    def __post_init__(self) -> None:
        _require_text(self.name, "initial value name")
        _require_text(self.units, f"units for initial value {self.name}")
        _require_finite(self.value, f"initial value {self.name}")


@dataclass(frozen=True, slots=True)
class InitialCondition:
    """Weighted initial condition available to a scenario episode."""

    name: str
    values: tuple[InitialValue, ...]
    weight: float = 1.0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.name, "initial condition name")
        if not self.values:
            raise ValueError(f"initial condition {self.name} requires at least one value")
        if self.weight <= 0.0:
            raise ValueError("initial condition weight must be positive")
        _require_finite(self.weight, f"weight for initial condition {self.name}")
        _require_unique(tuple(value.name for value in self.values), "initial value names")
        _validate_tags(self.tags)


@dataclass(frozen=True, slots=True)
class SignalTrajectory:
    """One value per control step for an exogenous signal."""

    signal: SignalSpec
    values: Array

    def __post_init__(self) -> None:
        values = jnp.asarray(self.values)
        if values.ndim != 1:
            raise ValueError(f"trajectory {self.signal.name} values must be one-dimensional")
        if not jnp.issubdtype(values.dtype, jnp.number):
            raise ValueError(f"trajectory {self.signal.name} values must be numeric")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError(f"trajectory {self.signal.name} values must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class ForecastTrajectory:
    """A trajectory that becomes controller-visible at a declared step."""

    trajectory: SignalTrajectory
    available_from_step: int = 0

    def __post_init__(self) -> None:
        if self.available_from_step < 0:
            raise ValueError("forecast availability step must be non-negative")


class EventKind(StrEnum):
    """Scenario mechanisms applied by the environment assembly."""

    DISTURBANCE = "disturbance"
    PARAMETER_DRIFT = "parameter_drift"
    EQUIPMENT_FAULT = "equipment_fault"
    SENSOR_FAULT = "sensor_fault"
    OPERATING = "operating"


class EventSeverity(StrEnum):
    """Comparable qualitative event severity."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(frozen=True, slots=True)
class TimedEvent:
    """Half-open event window, optionally repeated at a fixed frequency."""

    name: str
    kind: EventKind
    target: str
    units: str
    start_step: int
    end_step: int
    magnitude: float
    severity: EventSeverity
    tags: tuple[str, ...] = ()
    repeat_every_steps: int | None = None
    repeat_count: int = 1

    def __post_init__(self) -> None:
        _require_text(self.name, "event name")
        _require_text(self.target, f"target for event {self.name}")
        _require_text(self.units, f"units for event {self.name}")
        if self.start_step < 0:
            raise ValueError(f"event {self.name} start_step must be non-negative")
        if self.end_step <= self.start_step:
            raise ValueError(f"event {self.name} end_step must be after start_step")
        if self.repeat_count <= 0:
            raise ValueError(f"event {self.name} repeat_count must be positive")
        if self.repeat_every_steps is None and self.repeat_count != 1:
            raise ValueError(f"event {self.name} repeat_count requires repeat_every_steps")
        if self.repeat_every_steps is not None:
            if self.repeat_every_steps <= 0:
                raise ValueError(f"event {self.name} repeat_every_steps must be positive")
            if self.repeat_every_steps < self.duration_steps:
                raise ValueError(f"event {self.name} repetitions cannot overlap")
        _require_finite(self.magnitude, f"magnitude for event {self.name}")
        _validate_tags(self.tags)

    @property
    def duration_steps(self) -> int:
        """Return the duration of one occurrence."""
        return self.end_step - self.start_step

    @property
    def final_end_step(self) -> int:
        """Return the exclusive end of the final occurrence."""
        period = self.repeat_every_steps or 0
        return self.end_step + period * (self.repeat_count - 1)

    def is_active(self, step: int) -> bool:
        """Return whether an occurrence is active at this control step."""
        if step < self.start_step or step >= self.final_end_step:
            return False
        if self.repeat_every_steps is None:
            return step < self.end_step
        occurrence = (step - self.start_step) // self.repeat_every_steps
        offset = (step - self.start_step) % self.repeat_every_steps
        return occurrence < self.repeat_count and offset < self.duration_steps


def _validate_tags(tags: tuple[str, ...]) -> None:
    for tag in tags:
        _require_text(tag, "tag")
    _require_unique(tags, "tags")


@dataclass(frozen=True, slots=True)
class ScenarioEpisode:
    """Deterministic realization of a scenario pack for one seed."""

    spec: ScenarioSpec
    horizon_steps: int
    simulation_time_step: float
    signals: tuple[SignalSpec, ...]
    seeds: SeedBundle
    initial_condition: InitialCondition
    exogenous_signals: tuple[SignalTrajectory, ...]
    events: tuple[TimedEvent, ...]
    forecasts: tuple[ForecastTrajectory, ...]
    severity: EventSeverity
    tags: tuple[str, ...]

    def exogenous_at(self, step: int) -> dict[str, Array]:
        """Return all exogenous values at a control step."""
        self._validate_step(step)
        return {trajectory.signal.name: trajectory.values[step] for trajectory in self.exogenous_signals}

    def initial_state(self) -> dict[str, float]:
        """Return the selected initial condition as a name-to-value mapping."""
        return {value.name: value.value for value in self.initial_condition.values}

    def active_events(self, step: int) -> tuple[TimedEvent, ...]:
        """Return events active at a control step in stable schedule order."""
        self._validate_step(step)
        return tuple(event for event in self.events if event.is_active(step))

    def available_forecasts(self, step: int) -> tuple[ForecastTrajectory, ...]:
        """Return forecasts available to instrumentation at a control step."""
        self._validate_step(step)
        return tuple(forecast for forecast in self.forecasts if step >= forecast.available_from_step)

    def _validate_step(self, step: int) -> None:
        if step < 0 or step >= self.horizon_steps:
            raise IndexError(f"step {step} is outside episode horizon")


@dataclass(frozen=True, slots=True)
class ScenarioPack:
    """Validated scenario definition that can instantiate seeded episodes."""

    spec: ScenarioSpec
    horizon_steps: int
    simulation_time_step: float
    signals: tuple[SignalSpec, ...]
    initial_conditions: tuple[InitialCondition, ...]
    exogenous_signals: tuple[SignalTrajectory, ...] = ()
    events: tuple[TimedEvent, ...] = ()
    forecasts: tuple[ForecastTrajectory, ...] = ()
    severity: EventSeverity = EventSeverity.MODERATE
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("scenario horizon_steps must be positive")
        if self.simulation_time_step <= 0.0:
            raise ValueError("scenario simulation_time_step must be positive")
        _require_finite(self.simulation_time_step, "scenario simulation_time_step")
        if not self.initial_conditions:
            raise ValueError("scenario requires at least one initial condition")
        _validate_tags(self.tags)
        _require_unique(tuple(signal.name for signal in self.signals), "scenario signal names")
        _require_unique(
            tuple(condition.name for condition in self.initial_conditions),
            "initial condition names",
        )
        initial_schema = {value.name: value.units for value in self.initial_conditions[0].values}
        for condition in self.initial_conditions[1:]:
            condition_schema = {value.name: value.units for value in condition.values}
            if condition_schema != initial_schema:
                raise ValueError("initial conditions must share one value schema")
        _require_unique(
            tuple(trajectory.signal.name for trajectory in self.exogenous_signals),
            "exogenous trajectory names",
        )
        _require_unique(
            tuple(forecast.trajectory.signal.name for forecast in self.forecasts),
            "forecast trajectory names",
        )

        signal_units = {signal.name: signal.units for signal in self.signals}
        for trajectory in self.exogenous_signals:
            self._validate_trajectory(trajectory, signal_units)
        for forecast in self.forecasts:
            self._validate_trajectory(forecast.trajectory, signal_units)
            if forecast.available_from_step >= self.horizon_steps:
                raise ValueError("forecast availability must be inside the scenario horizon")

        ordered_events = tuple(sorted(self.events, key=lambda event: (event.start_step, event.name)))
        if ordered_events != self.events:
            raise ValueError("scenario events must be ordered by start_step then name")
        _require_unique(tuple(event.name for event in self.events), "scenario event names")
        for event in self.events:
            if event.final_end_step > self.horizon_steps:
                raise ValueError(f"event {event.name} extends beyond the scenario horizon")
            if event.kind in (EventKind.DISTURBANCE, EventKind.SENSOR_FAULT):
                expected_units = signal_units.get(event.target)
                if expected_units is None:
                    raise ValueError(f"event {event.name} targets unknown signal {event.target}")
                if event.units != expected_units:
                    raise ValueError(f"event {event.name} units {event.units!r} do not match signal units {expected_units!r}")

    def _validate_trajectory(self, trajectory: SignalTrajectory, signal_units: dict[str, str]) -> None:
        expected_units = signal_units.get(trajectory.signal.name)
        if expected_units is None:
            raise ValueError(f"trajectory targets unknown signal {trajectory.signal.name}")
        if trajectory.signal.units != expected_units:
            raise ValueError(f"trajectory {trajectory.signal.name} units {trajectory.signal.units!r} do not match scenario units {expected_units!r}")
        if trajectory.values.shape[0] != self.horizon_steps:
            raise ValueError(f"trajectory {trajectory.signal.name} length must match scenario horizon")

    def instantiate(self, seed: int) -> ScenarioEpisode:
        """Select initial conditions and return a reproducible episode."""
        seeds = SeedBundle.from_seed(seed)
        weights = jnp.asarray(tuple(condition.weight for condition in self.initial_conditions))
        probabilities = weights / jnp.sum(weights)
        selected = int(jax.random.choice(seeds.key(RandomStream.SCENARIO), len(self.initial_conditions), p=probabilities))
        return ScenarioEpisode(
            spec=self.spec,
            horizon_steps=self.horizon_steps,
            simulation_time_step=self.simulation_time_step,
            signals=self.signals,
            seeds=seeds,
            initial_condition=self.initial_conditions[selected],
            exogenous_signals=self.exogenous_signals,
            events=self.events,
            forecasts=self.forecasts,
            severity=self.severity,
            tags=self.tags,
        )


__all__ = [
    "EventKind",
    "EventSeverity",
    "ForecastTrajectory",
    "InitialCondition",
    "InitialValue",
    "RandomStream",
    "ScenarioEpisode",
    "ScenarioPack",
    "SeedBundle",
    "SignalTrajectory",
    "TimedEvent",
]
