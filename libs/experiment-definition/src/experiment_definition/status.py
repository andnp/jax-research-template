"""Execution status vocabulary shared by the schema, planner and readers."""

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


LIVE = frozenset({ExecutionStatus.PENDING, ExecutionStatus.RUNNING})
# A run with an execution in any of these is not planned again.
CLAIMING = LIVE | {ExecutionStatus.COMPLETED}
FINISHED = frozenset({ExecutionStatus.COMPLETED, ExecutionStatus.FAILED})


def sql_in(statuses: frozenset[ExecutionStatus]) -> str:
    return "(" + ", ".join(f"'{status}'" for status in ExecutionStatus if status in statuses) + ")"


class RunState(StrEnum):
    """Outcome of a run's execution attempts."""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


def run_state(statuses: frozenset[str]):
    if ExecutionStatus.COMPLETED in statuses:
        return RunState.COMPLETED
    # A live execution outranks a failed one, since the run may yet complete.
    if ExecutionStatus.FAILED in statuses and not statuses & LIVE:
        return RunState.FAILED
    return RunState.PENDING
