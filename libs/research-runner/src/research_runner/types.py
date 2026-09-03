from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment: Experiment
    train_fn: Callable[[ExecutionContext], ExecutionResult]
    db_path: Path
    executions_root: Path
    metrics_db_path: Path | None = None
    max_runs_per_batch: int | None = None
    capture_git: bool = True
