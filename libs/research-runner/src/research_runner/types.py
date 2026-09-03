from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from experiment_definition import Experiment
from experiment_definition.db import ExperimentRow, RunRow


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    execution_id: int
    experiment: ExperimentRow
    runs: list[RunRow]
    hyperparameters: dict[str, object]
    seed_values: tuple[int, ...]
    execution_root: Path
    metrics_db_path: Path


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
