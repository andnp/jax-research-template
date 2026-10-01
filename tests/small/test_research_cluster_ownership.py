from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence

import pytest
from experiment_definition.ownership import OwnerState, local_owner
from research_cluster.ownership import (
    format_slurm_owner,
    owner_state,
    parse_job_states,
    parse_slurm_owner,
)


def _tools(**outputs: str | Exception):
    """A command runner answering each Slurm binary with a canned output or error."""

    def run(command: Sequence[str]) -> str:
        answer = outputs[command[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return run


def _failed(stderr: str):
    return subprocess.CalledProcessError(1, "squeue", stderr=stderr)


class TestSlurmOwnerFormat:
    def test_formatted_owner_parses_back_to_its_task(self) -> None:
        """The task id in a formatted owner is the ``<job>_<index>`` Slurm tools accept."""
        assert parse_slurm_owner(format_slurm_owner("4812", 3)) == "4812_3"

    @pytest.mark.parametrize("owner", ["local:h:b:1:2", "slurm:4812", "slurm:x_1", "slurm:4812_", ""])
    def test_other_owners_are_not_slurm_owners(self, owner: str) -> None:
        """Anything but ``slurm:<digits>_<digits>`` is rejected."""
        assert parse_slurm_owner(owner) is None


class TestParseJobStates:
    def test_strips_padding_blank_lines_and_cancel_suffixes(self) -> None:
        """Only the first word of each line is a state; ``CANCELLED by 1000`` and ``CANCELLED+`` both become CANCELLED."""
        output = "RUNNING  \n\n CANCELLED by 1000\nCANCELLED+\n"
        assert parse_job_states(output) == ["RUNNING", "CANCELLED", "CANCELLED"]


class TestSlurmOwnerState:
    @pytest.mark.parametrize("queued", ["RUNNING\n", "PENDING\n", "COMPLETING\n"])
    def test_queued_task_is_alive(self, queued: str) -> None:
        """A non-terminal state in the queue means the task will still finish."""
        run = _tools(squeue=queued, sacct=AssertionError("sacct is not consulted"))
        assert owner_state("slurm:7_0", run) is OwnerState.ALIVE

    @pytest.mark.parametrize(
        "accounted",
        ["COMPLETED\n", "FAILED\n", "CANCELLED by 1000\n", "TIMEOUT\n", "OUT_OF_MEMORY\n", "NODE_FAIL\n", "PREEMPTED\n"],
    )
    def test_task_that_left_the_queue_in_a_terminal_state_is_dead(self, accounted: str) -> None:
        """sacct reporting a terminal state means the task is over."""
        assert owner_state("slurm:7_0", _tools(squeue="", sacct=accounted)) is OwnerState.DEAD

    def test_task_unknown_to_both_tools_is_dead(self) -> None:
        """A task absent from the queue and the accounting history is gone."""
        invalid = _failed("slurm_load_jobs error: Invalid job id specified")
        assert owner_state("slurm:7_0", _tools(squeue=invalid, sacct="")) is OwnerState.DEAD

    def test_task_still_running_in_accounting_is_alive(self) -> None:
        """A queue that has not caught up with sacct does not make a running task dead."""
        assert owner_state("slurm:7_0", _tools(squeue="", sacct="RUNNING\n")) is OwnerState.ALIVE

    @pytest.mark.parametrize("missing", ["squeue", "sacct"])
    def test_missing_binary_is_unknown(self, missing: str) -> None:
        """Without a Slurm client the owner cannot be judged."""
        tools: Mapping[str, str | Exception] = {"squeue": "", "sacct": "", missing: FileNotFoundError(missing)}
        assert owner_state("slurm:7_0", _tools(**tools)) is OwnerState.UNKNOWN

    def test_failing_tool_is_unknown(self) -> None:
        """A Slurm controller error is not evidence that the task died."""
        run = _tools(squeue=_failed("slurm_load_jobs error: Unable to contact slurm controller"))
        assert owner_state("slurm:7_0", run) is OwnerState.UNKNOWN


class TestOtherOwners:
    def test_local_owner_is_judged_by_the_local_host(self) -> None:
        """Non-Slurm owners never touch Slurm tools."""
        run = _tools()
        assert owner_state(local_owner(), run) is OwnerState.ALIVE

    def test_unrecognised_owner_is_unknown(self) -> None:
        """An owner nobody can interpret is left alone."""
        assert owner_state("k8s:pod-1", _tools()) is OwnerState.UNKNOWN
