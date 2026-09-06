"""Public API for the research-runner library."""

from __future__ import annotations

from .registry import registered_specs, spec
from .runner import execute_batch, run_experiment
from .types import (
    BatchTrainer,
    BoundBatch,
    ExecutionContext,
    ExecutionResult,
    ExperimentSpec,
    ParameterSchema,
    RunPoint,
    get_parameter,
)

__all__ = [
    "BatchTrainer",
    "BoundBatch",
    "ExecutionContext",
    "ExecutionResult",
    "ExperimentSpec",
    "ParameterSchema",
    "RunPoint",
    "execute_batch",
    "get_parameter",
    "registered_specs",
    "run_experiment",
    "spec",
]
