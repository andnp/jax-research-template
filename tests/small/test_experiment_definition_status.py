"""Contract tests for execution status transitions."""

from pathlib import Path

import pytest
from experiment_definition.db import DatabaseManager
from experiment_definition.status import ExecutionStatus

PENDING, RUNNING, COMPLETED, FAILED, INVALID = ExecutionStatus


def _execution_at(database: DatabaseManager, path: tuple[ExecutionStatus, ...]):
    execution_id = database.add_execution()
    for status in path:
        database.update_execution_status(execution_id, status)
    return execution_id


@pytest.fixture
def database(tmp_path: Path):
    with DatabaseManager(tmp_path / "experiments.sqlite") as database:
        database.initialize()
        yield database


class TestTransitions:
    @pytest.mark.parametrize(
        ("path", "target"),
        [
            ((), RUNNING),
            ((), FAILED),
            ((RUNNING,), FAILED),
            ((RUNNING,), COMPLETED),
            ((RUNNING, COMPLETED), INVALID),
            ((RUNNING, FAILED), INVALID),
        ],
    )
    def test_lifecycle_and_recovery_writes_are_allowed(
        self, database: DatabaseManager, path: tuple[ExecutionStatus, ...], target: ExecutionStatus
    ) -> None:
        """Claiming, finishing, failing a live execution, and invalidating a finished one all succeed.

        Stale reclaim, the dead-worker sweep, sbatch failure and commit invalidation rely on these.
        """
        execution_id = _execution_at(database, path)

        database.update_execution_status(execution_id, target)

        execution = database.get_execution(execution_id)
        assert execution is not None
        assert execution.status == target

    @pytest.mark.parametrize(
        ("path", "target"),
        [
            ((RUNNING,), PENDING),
            ((RUNNING, COMPLETED), RUNNING),
            ((RUNNING, FAILED), COMPLETED),
            ((INVALID,), PENDING),
        ],
    )
    def test_rewinds_are_rejected(
        self, database: DatabaseManager, path: tuple[ExecutionStatus, ...], target: ExecutionStatus
    ) -> None:
        """A finished or claimed execution never returns to an earlier state, so results cannot be overwritten."""
        execution_id = _execution_at(database, path)

        with pytest.raises(ValueError, match="Illegal execution transition"):
            database.update_execution_status(execution_id, target)


class TestFailExecution:
    @pytest.mark.parametrize("path", [(), (RUNNING,)])
    def test_live_execution_fails_with_an_end_time(
        self, database: DatabaseManager, path: tuple[ExecutionStatus, ...]
    ) -> None:
        """A pending or running execution becomes FAILED and records when it ended."""
        execution_id = _execution_at(database, path)

        database.fail_execution(execution_id)

        execution = database.get_execution(execution_id)
        assert execution is not None
        assert execution.status == FAILED
        assert execution.end_time is not None

    @pytest.mark.parametrize("path", [(RUNNING, COMPLETED), (RUNNING, FAILED), (INVALID,)])
    def test_finished_execution_is_left_alone(
        self, database: DatabaseManager, path: tuple[ExecutionStatus, ...]
    ) -> None:
        """A failure handler racing a finished execution must not overwrite its outcome."""
        execution_id = _execution_at(database, path)

        database.fail_execution(execution_id)

        execution = database.get_execution(execution_id)
        assert execution is not None
        assert execution.status == path[-1]
        assert execution.end_time is None
