"""Slurm array tasks as execution owners (ADR 014)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

from experiment_definition.ownership import OwnerState, local_owner_state

type CommandRunner = Callable[[Sequence[str]], str]

_SLURM_PREFIX = "slurm"
_TERMINAL_STATES = frozenset({
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
})
_UNKNOWN_JOB_MESSAGE = "Invalid job id"


def format_slurm_owner(array_job_id: str, task_index: int) -> str:
    return f"{_SLURM_PREFIX}:{array_job_id}_{task_index}"


def parse_slurm_owner(owner: str) -> str | None:
    """Return the ``<array_job_id>_<task_index>`` of a ``slurm:`` owner, or None when ``owner`` is not one."""
    prefix, _, task = owner.partition(":")
    job_id, _, index = task.partition("_")
    if prefix != _SLURM_PREFIX or not (job_id.isdigit() and index.isdigit()):
        return None
    return task


def parse_job_states(output: str) -> list[str]:
    """Read job states from ``squeue``/``sacct`` output, dropping suffixes such as ``CANCELLED by 1000`` and ``CANCELLED+``."""
    return [line.split()[0].rstrip("+") for line in output.splitlines() if line.strip()]


def run_command(command: Sequence[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def _queued_states(task: str, run: CommandRunner) -> list[str]:
    try:
        return parse_job_states(run(["squeue", "-h", "-j", task, "-o", "%T"]))
    except subprocess.CalledProcessError as error:
        if _UNKNOWN_JOB_MESSAGE in (error.stderr or ""):
            return []
        raise


def _task_state(task: str, run: CommandRunner) -> OwnerState:
    states = _queued_states(task, run)
    if not states:
        states = parse_job_states(run(["sacct", "-n", "-X", "-j", task, "-o", "State"]))
    return OwnerState.DEAD if all(state in _TERMINAL_STATES for state in states) else OwnerState.ALIVE


def owner_state(owner: str, run: CommandRunner = run_command) -> OwnerState:
    """Report whether an owner is still running: ``slurm:`` owners are asked of Slurm, any other owner of the local host.

    A task absent from both ``squeue`` and ``sacct`` is DEAD; unreachable or failing Slurm tools give UNKNOWN.
    """
    task = parse_slurm_owner(owner)
    if task is None:
        return local_owner_state(owner)
    try:
        return _task_state(task, run)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return OwnerState.UNKNOWN
