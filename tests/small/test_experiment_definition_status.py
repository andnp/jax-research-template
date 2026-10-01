"""Contract tests for execution status transitions and derived run states."""

from pathlib import Path

import pytest
from experiment_definition._execution_repository import _transition_many
from experiment_definition.db import DatabaseManager
from experiment_definition.status import ExecutionStatus, RunState, run_state

PENDING, RUNNING, COMPLETED, FAILED, INVALID = ExecutionStatus


def _execution_at(database: DatabaseManager, path: tuple[ExecutionStatus, ...]):
    execution_id = database.add_execution()
    for status in path:
        database.update_execution_status(execution_id, status)
    return execution_id


def _status_of(database: DatabaseManager, execution_id: int):
    execution = database.get_execution(execution_id)
    assert execution is not None
    return execution.status


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


class TestTransitionMany:
    def test_transitions_every_eligible_execution_and_skips_the_rest(self, database: DatabaseManager) -> None:
        """Missing, finished and duplicate ids are skipped; only executions that legally fail are returned."""
        pending = _execution_at(database, ())
        running = _execution_at(database, (RUNNING,))
        completed = _execution_at(database, (RUNNING, COMPLETED))

        failed = _transition_many(database.conn, [pending, completed, 999999, running, pending], FAILED, "t")

        assert failed == [pending, running]
        assert _status_of(database, pending) == FAILED
        assert _status_of(database, completed) == COMPLETED

    def test_rolls_back_every_transition_when_one_fails(self, database: DatabaseManager) -> None:
        """A write error part-way through leaves no execution half-transitioned."""
        first = _execution_at(database, ())
        second = _execution_at(database, ())
        database.conn.execute(
            f"CREATE TRIGGER block_second BEFORE UPDATE ON Executions WHEN OLD.id = {second} "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )

        with pytest.raises(Exception, match="blocked"):
            _transition_many(database.conn, [first, second], FAILED)

        assert _status_of(database, first) == PENDING


class TestFailExecutions:
    def test_fails_only_the_live_executions_and_returns_them(self, database: DatabaseManager) -> None:
        """Pending and running executions fail together; finished and missing ids are left out."""
        pending = _execution_at(database, ())
        running = _execution_at(database, (RUNNING,))
        completed = _execution_at(database, (RUNNING, COMPLETED))

        failed = database.fail_executions([pending, running, completed, 999999])

        assert failed == [pending, running]
        assert [_status_of(database, i) for i in (pending, running, completed)] == [FAILED, FAILED, COMPLETED]


class TestInvalidateExecutions:
    def test_invalidates_every_existing_execution_once_and_returns_them(self, database: DatabaseManager) -> None:
        """Executions in any state become INVALID; missing, already invalid and repeated ids are skipped."""
        pending = _execution_at(database, ())
        completed = _execution_at(database, (RUNNING, COMPLETED))
        invalid = _execution_at(database, (INVALID,))

        invalidated = database.invalidate_executions([completed, 999999, invalid, pending, completed])

        assert invalidated == [completed, pending]
        assert [_status_of(database, i) for i in (pending, completed, invalid)] == [INVALID] * 3


class TestRunState:
    @pytest.mark.parametrize(
        ("statuses", "expected"),
        [
            (frozenset(), RunState.PENDING),
            (frozenset({PENDING}), RunState.PENDING),
            (frozenset({INVALID}), RunState.PENDING),
            (frozenset({FAILED}), RunState.FAILED),
            (frozenset({FAILED, INVALID}), RunState.FAILED),
            (frozenset({FAILED, RUNNING}), RunState.PENDING),
            (frozenset({FAILED, PENDING}), RunState.PENDING),
            (frozenset({FAILED, COMPLETED}), RunState.COMPLETED),
            (frozenset({RUNNING, COMPLETED}), RunState.COMPLETED),
        ],
    )
    def test_precedence(self, statuses: frozenset[str], expected: RunState) -> None:
        """Any completion wins, a live attempt outranks a failure, and invalidated attempts count for nothing.

        A run whose only attempts failed must stay distinguishable from one never attempted.
        """
        assert run_state(statuses) is expected
