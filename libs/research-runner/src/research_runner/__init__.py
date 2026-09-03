"""Public API for the research-runner library."""

from __future__ import annotations

from .registry import registered_specs, spec
from .runner import execute_batch, run_experiment
from .types import ExecutionContext, ExecutionResult, ExperimentSpec, RunPoint

__all__ = [
    "ExecutionContext",
    "ExecutionResult",
    "ExperimentSpec",
    "RunPoint",
    "execute_batch",
    "registered_specs",
    "run_experiment",
    "spec",
]
