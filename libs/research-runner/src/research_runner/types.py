from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
