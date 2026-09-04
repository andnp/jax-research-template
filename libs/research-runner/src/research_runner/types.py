from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar, cast, runtime_checkable

from experiment_definition import Experiment, ParameterValue


@dataclass(frozen=True, slots=True)
class RunPoint:
    """One logical run's resolved hyperparameters, including its seed."""

    run_id: int
    hyperparameters: Mapping[str, ParameterValue]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    execution_id: int
    experiment_id: int
    experiment_name: str
    static_config: Mapping[str, ParameterValue]
    points: tuple[RunPoint, ...]
    execution_root: Path
    metrics_db_path: Path

    def bind(self, schema: "ParameterSchema") -> "BoundBatch":
        return BoundBatch.from_context(self, schema)

    def axes(self) -> dict[str, list[ParameterValue]]:
        """Parameters that vary across the batch, stacked in `points` order.

        Only parameters carried by every point are returned, so each list is a
        usable vmap axis.
        """
        if not self.points:
            return {}
        shared = set.intersection(*(set(point.hyperparameters) for point in self.points))
        return {
            key: [point.hyperparameters[key] for point in self.points]
            for key in sorted(shared - set(self.static_config))
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    metadata: dict[str, object] = field(default_factory=dict)
    metric_curves: Mapping[str, Sequence[Sequence[float]]] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """Name the hyperparameters that a trainer can batch dynamically."""

    dynamic_keys: frozenset[str] = frozenset()


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class BoundBatch:
    """Validated static and dynamic values for one trainer execution."""

    static_config: Mapping[str, ParameterValue]
    dynamic_axes: Mapping[str, tuple[ParameterValue, ...]]
    seeds: tuple[int, ...]
    points: tuple[RunPoint, ...]

    @classmethod
    def from_context(cls, context: ExecutionContext, schema: ParameterSchema) -> "BoundBatch":
        axes = context.axes()
        seed_axis = axes.pop("seed", None)
        if seed_axis is None:
            raise ValueError("every execution point must provide a seed")
        seeds = tuple(_coerce(value, int, "seed") for value in seed_axis)
        if len(seeds) != len(context.points):
            raise ValueError("the seed axis must match the number of execution points")
        unknown = set(axes) - schema.dynamic_keys
        if unknown:
            raise ValueError(f"dynamic parameters are not supported by this trainer: {sorted(unknown)}")
        return cls(
            static_config=context.static_config,
            dynamic_axes={key: tuple(values) for key, values in axes.items()},
            seeds=seeds,
            points=context.points,
        )

    def static(self, name: str, expected_type: type[_T]) -> _T:
        try:
            value = self.static_config[name]
        except KeyError as error:
            raise ValueError(f"missing static parameter {name!r}") from error
        return _coerce(value, expected_type, name)

    def static_or(self, name: str, expected_type: type[_T], default: _T) -> _T:
        if name not in self.static_config:
            return default
        return self.static(name, expected_type)

    def dynamic(self, name: str, expected_type: type[_T]) -> tuple[_T, ...]:
        try:
            values = self.dynamic_axes[name]
        except KeyError as error:
            raise ValueError(f"missing dynamic parameter {name!r}") from error
        return tuple(_coerce(value, expected_type, name) for value in values)


def _coerce(value: ParameterValue, expected_type: type[_T], name: str) -> _T:
    if expected_type is float and isinstance(value, int | float) and not isinstance(value, bool):
        return cast(_T, float(value))
    if expected_type is int and isinstance(value, int) and not isinstance(value, bool):
        return cast(_T, value)
    if isinstance(value, expected_type):
        return cast(_T, value)
    raise TypeError(f"parameter {name!r} must be {expected_type.__name__}, got {value!r}")


@runtime_checkable
class BatchTrainer(Protocol):
    @property
    def parameter_schema(self) -> ParameterSchema: ...

    def __call__(self, context: ExecutionContext) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A runnable experiment and the results tree its artifacts live under."""

    experiment: Experiment
    train_fn: Callable[[ExecutionContext], ExecutionResult]
    results_root: Path
    max_runs_per_batch: int | None = None
    metric_subsample_factor: int = 1
    capture_git: bool = True

    def __post_init__(self) -> None:
        if self.metric_subsample_factor < 1:
            raise ValueError("metric_subsample_factor must be at least 1")
        if not self.results_root.is_absolute():
            raise ValueError(
                f"results_root must be absolute, got {self.results_root}. "
                "A relative path resolves against the caller's working directory, "
                "which silently moves the results tree.",
            )

    @property
    def parameter_schema(self) -> ParameterSchema | None:
        if isinstance(self.train_fn, BatchTrainer):
            return self.train_fn.parameter_schema
        return None

    @property
    def static_parameter_names(self) -> frozenset[str] | None:
        schema = self.parameter_schema
        if schema is None:
            return None
        return frozenset(
            name
            for name in self.experiment.parameter_names
            if name != "seed" and name not in schema.dynamic_keys
        )

    def sync(self, db_path: Path | str):
        return self.experiment.sync(db_path, static_parameter_names=self.static_parameter_names)

    @property
    def db_path(self) -> Path:
        return self.results_root / "experiments.sqlite"

    @property
    def metrics_db_path(self) -> Path:
        return self.results_root / "metrics.sqlite"

    @property
    def executions_root(self) -> Path:
        return self.results_root / "executions"
