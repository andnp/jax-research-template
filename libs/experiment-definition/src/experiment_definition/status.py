"""Execution status vocabulary shared by the schema, planner and readers."""

from collections.abc import Iterable
from enum import StrEnum


class ExecutionStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALID = "INVALID"


TRANSITIONS: dict[ExecutionStatus, frozenset[ExecutionStatus]] = {
    ExecutionStatus.PENDING: frozenset(
        {ExecutionStatus.RUNNING, ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.INVALID}
    ),
    ExecutionStatus.RUNNING: frozenset({ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.INVALID}),
    ExecutionStatus.COMPLETED: frozenset({ExecutionStatus.INVALID}),
    ExecutionStatus.FAILED: frozenset({ExecutionStatus.INVALID}),
    ExecutionStatus.INVALID: frozenset(),
}


def sql_in(statuses: Iterable[ExecutionStatus]) -> str:
    return "(" + ", ".join(f"'{status}'" for status in statuses) + ")"
