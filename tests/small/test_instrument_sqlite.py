"""Small tests for research_instrument.sqlite_backend — persistence layer."""

import math
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest
from research_instrument.scalar import write_execution_scalars, write_seed_batch_scalars
from research_instrument.seed_batch import write_seed_batch_curve
from research_instrument.sqlite_backend import open_metrics_connection


def write_curve(
    db_path: Path,
    values: list[float],
    *,
    metric_name: str = "reward",
    run_id: int = 101,
    execution_id: int = 1001,
    seed_id: int = 0,
) -> None:
    write_seed_batch_curve(
        db_path,
        [values],
        metric_name=metric_name,
        experiment_id=10,
        execution_id=execution_id,
        run_ids=[run_id],
        seed_ids=[seed_id],
    )


_NOT_NULL_VALUE_TABLE_SQL = (
    "CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT NOT NULL, "
    "value REAL NOT NULL, global_step INTEGER NOT NULL, seed_id INTEGER NOT NULL, "
    "experiment_id INTEGER NOT NULL, run_id INTEGER NOT NULL, execution_id INTEGER NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)


class TestMetricsSchema:
    def test_executions_can_share_one_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "shared.db"
            write_curve(db_path, [1.0], run_id=101, execution_id=1001)
            write_curve(db_path, [2.0], run_id=102, execution_id=1002)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT experiment_id, run_id, execution_id, metric_name, global_step, value "
                    "FROM metrics ORDER BY execution_id"
                ).fetchall()
                assert rows == [
                    (10, 101, 1001, "reward", 0, 1.0),
                    (10, 102, 1002, "reward", 0, 2.0),
                ]

                index_names = {
                    row[1]
                    for row in conn.execute("PRAGMA index_list(metrics)").fetchall()
                }
                assert "idx_metrics_experiment_metric_name_step" in index_names
                assert "idx_metrics_run_metric_name_step" in index_names
                assert "idx_metrics_execution_metric_name_step" in index_names

    def test_existing_table_is_migrated_to_metric_name_when_identity_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        global_step INTEGER NOT NULL,
                        seed_id INTEGER NOT NULL,
                        experiment_id INTEGER,
                        run_id INTEGER,
                        execution_id INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX idx_metrics_name ON metrics (name);
                    CREATE INDEX idx_metrics_step ON metrics (global_step);
                    CREATE INDEX idx_metrics_name_seed ON metrics (name, seed_id);
                    """
                )
                conn.execute(
                    "INSERT INTO metrics (name, value, global_step, seed_id, experiment_id, run_id, execution_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("reward", 2.5, 6, 2, 10, 101, 1001),
                )

            write_curve(db_path, [0.0] * 7 + [3.0], seed_id=2)

            with sqlite3.connect(db_path) as conn:
                columns = {
                    row[1]: bool(row[3])
                    for row in conn.execute("PRAGMA table_info(metrics)").fetchall()
                }
                assert "name" not in columns
                assert columns["metric_name"] is True
                assert columns["experiment_id"] is True
                assert columns["run_id"] is True
                assert columns["execution_id"] is True

                stored_rows = conn.execute(
                    "SELECT experiment_id, run_id, execution_id, metric_name, global_step, value "
                    "FROM metrics WHERE global_step IN (6, 7) ORDER BY global_step, value"
                ).fetchall()
                assert stored_rows == [
                    (10, 101, 1001, "reward", 6, 0.0),
                    (10, 101, 1001, "reward", 6, 2.5),
                    (10, 101, 1001, "reward", 7, 3.0),
                ]

    def test_not_null_value_table_is_migrated_to_accept_nan(self) -> None:
        """A database created before NaN support keeps its rows and records a diverged value after reopening."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "strict.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(_NOT_NULL_VALUE_TABLE_SQL)
                conn.execute(
                    "INSERT INTO metrics (metric_name, value, global_step, seed_id, experiment_id, run_id, execution_id) "
                    "VALUES ('loss', 0.5, 0, 0, 10, 101, 1001)"
                )

            write_curve(db_path, [math.nan], metric_name="loss", seed_id=0)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT value FROM metrics ORDER BY id").fetchall()
            assert rows[0] == (0.5,)
            assert rows[1] == (None,)

    def test_journal_mode_is_not_wal(self) -> None:
        """WAL requires its wal-index in host-local shared memory and does not
        work over the network filesystems (NFS/Lustre) Slurm array tasks share
        metrics.sqlite over, so the connection must stay on the default
        rollback journal rather than switching to WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "journal.db"
            write_curve(db_path, [1.0])

            with sqlite3.connect(db_path) as conn:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

            assert str(mode).lower() != "wal"

    def test_wal_database_from_an_earlier_version_is_converted_to_rollback(self) -> None:
        """Journal mode is persisted in the file, so a metrics DB created while WAL was
        the default (commit 63c8bec) stays WAL until something converts it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy_wal.db"
            setup_conn = sqlite3.connect(db_path)
            setup_conn.execute("PRAGMA journal_mode=WAL")
            setup_conn.execute(
                "CREATE TABLE metrics ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT NOT NULL, value REAL, "
                "global_step INTEGER NOT NULL, seed_id INTEGER NOT NULL, experiment_id INTEGER NOT NULL, "
                "run_id INTEGER NOT NULL, execution_id INTEGER NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            setup_conn.commit()
            mode = setup_conn.execute("PRAGMA journal_mode").fetchone()[0]
            setup_conn.close()
            assert str(mode).lower() == "wal"

            conn = open_metrics_connection(db_path)
            try:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            finally:
                conn.close()

            assert str(mode).lower() != "wal"

    def test_legacy_table_without_identity_columns_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        global_step INTEGER NOT NULL,
                        seed_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

            with pytest.raises(RuntimeError, match="legacy schema without required identity columns"):
                open_metrics_connection(db_path)


class TestConcurrentWriters:
    def test_two_connections_write_concurrently_without_database_locked_errors(self) -> None:
        """Two writers (simulating separate Slurm array tasks
        sharing one metrics.sqlite) both complete under contention instead of
        one raising 'database is locked'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "concurrent.db"
            errors: list[BaseException] = []

            def write_many(run_id: int, seed_id: int) -> None:
                try:
                    for step in range(200):
                        write_curve(db_path, [float(step)], run_id=run_id, execution_id=run_id, seed_id=seed_id)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=write_many, args=(101, 0)),
                threading.Thread(target=write_many, args=(102, 1)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert errors == []
            with sqlite3.connect(db_path) as conn:
                counts = dict(conn.execute("SELECT run_id, COUNT(*) FROM metrics GROUP BY run_id").fetchall())
            assert counts == {101: 200, 102: 200}


class TestWriteSeedBatchCurve:
    def test_each_row_lands_under_its_own_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seed_batch.db"
            curves = [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
            run_ids = [201, 202, 203]
            seed_ids = [1, 2, 3]

            write_seed_batch_curve(
                db_path,
                curves,
                metric_name="returned_episode_returns",
                experiment_id=10,
                execution_id=1001,
                run_ids=run_ids,
                seed_ids=seed_ids,
            )

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT run_id, seed_id, global_step, value FROM metrics ORDER BY run_id, global_step"
                ).fetchall()
            assert rows == [
                (201, 1, 0, 10.0),
                (201, 1, 1, 11.0),
                (202, 2, 0, 20.0),
                (202, 2, 1, 21.0),
                (203, 3, 0, 30.0),
                (203, 3, 1, 31.0),
            ]

    def test_mispaired_run_ids_attribute_wrong_seed_to_wrong_run(self) -> None:
        """Deliberately scramble run_ids relative to curves: rows land under the wrong run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seed_batch.db"
            curves = [[10.0], [20.0], [30.0]]
            correct_run_ids = [201, 202, 203]
            scrambled_run_ids = [202, 203, 201]  # row 0 (seed's value 10.0) mispaired to run 202
            seed_ids = [1, 2, 3]

            write_seed_batch_curve(
                db_path,
                curves,
                metric_name="returned_episode_returns",
                experiment_id=10,
                execution_id=1001,
                run_ids=scrambled_run_ids,
                seed_ids=seed_ids,
            )

            with sqlite3.connect(db_path) as conn:
                value_by_run = dict(
                    conn.execute("SELECT run_id, value FROM metrics").fetchall()
                )
            # Row 0's value (10.0) belongs to run 201 under correct pairing, but the
            # scrambled run_ids attribute it to run 202 instead — proving that a
            # mispairing between curves and run_ids silently mislabels results.
            assert value_by_run[correct_run_ids[0]] != 10.0
            assert value_by_run[scrambled_run_ids[0]] == 10.0

    def test_mismatched_lengths_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seed_batch.db"
            with pytest.raises(ValueError):
                write_seed_batch_curve(
                    db_path,
                    [[1.0], [2.0]],
                    metric_name="reward",
                    experiment_id=10,
                    execution_id=1001,
                    run_ids=[201],
                    seed_ids=[1, 2],
                )

    def test_subsample_preserves_true_step_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seed_batch.db"
            curve = [float(i) for i in range(20)]

            write_seed_batch_curve(
                db_path,
                [curve],
                metric_name="reward",
                experiment_id=10,
                execution_id=1001,
                run_ids=[301],
                seed_ids=[7],
                subsample_factor=5,
            )

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT global_step, value FROM metrics ORDER BY global_step"
                ).fetchall()
            # Steps are not renumbered 0..3 — the original step indices survive.
            assert rows == [(0, 0.0), (5, 5.0), (10, 10.0), (15, 15.0)]

    def test_nan_values_round_trip(self) -> None:
        """Diverged runs are recorded as NaN, not dropped, by the single-transaction write path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seed_batch.db"

            write_seed_batch_curve(
                db_path,
                [[1.0, math.nan]],
                metric_name="reward",
                experiment_id=10,
                execution_id=1001,
                run_ids=[301],
                seed_ids=[7],
            )

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT global_step, value FROM metrics ORDER BY global_step").fetchall()
            assert rows[0] == (0, 1.0)
            assert rows[1][1] is None  # SQLite stores NaN as NULL.

    def test_rows_spanning_multiple_batch_chunks_all_persist(self) -> None:
        """A batch_size smaller than the total row count still writes every row
        under the one connection/transaction, with schema DDL applied only once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seed_batch.db"
            curves = [[float(step) for step in range(5)] for _ in range(3)]
            run_ids = [201, 202, 203]
            seed_ids = [1, 2, 3]

            write_seed_batch_curve(
                db_path,
                curves,
                metric_name="reward",
                experiment_id=10,
                execution_id=1001,
                run_ids=run_ids,
                seed_ids=seed_ids,
                batch_size=4,
            )

            with sqlite3.connect(db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
            assert count == 15

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "seed_batch.db"
            write_seed_batch_curve(
                db_path,
                [[1.0]],
                metric_name="reward",
                experiment_id=10,
                execution_id=1001,
                run_ids=[301],
                seed_ids=[7],
            )
            assert db_path.exists()


class TestScalarMetricWriters:
    def test_writes_both_scopes_alongside_existing_curves(self, tmp_path: Path) -> None:
        """
        Verify scalar scopes persist with the right identities while curve rows remain intact.
        """
        db_path = tmp_path / "metrics.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "CREATE TABLE metrics ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT NOT NULL, value REAL, "
                "global_step INTEGER NOT NULL, seed_id INTEGER NOT NULL, experiment_id INTEGER NOT NULL, "
                "run_id INTEGER NOT NULL, execution_id INTEGER NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "INSERT INTO metrics (metric_name, value, global_step, seed_id, experiment_id, "
                "run_id, execution_id) "
                "VALUES ('legacy_curve', 4.0, 0, 1, 10, 201, 1001)"
            )

        write_seed_batch_scalars(
            db_path,
            {"score": (1.0, math.nan)},
            experiment_id=10,
            execution_id=1001,
            run_ids=(201, 202),
            seed_ids=(1, 2),
        )
        write_execution_scalars(
            db_path,
            {"batch_score": 1.5},
            experiment_id=10,
            execution_id=1001,
        )

        with sqlite3.connect(db_path) as connection:
            curve_rows = connection.execute(
                "SELECT metric_name, value, global_step, run_id FROM metrics"
            ).fetchall()
            scalar_rows = connection.execute(
                "SELECT scope, metric_name, value, run_id, seed_id, execution_id "
                "FROM scalar_metrics ORDER BY scope, run_id"
            ).fetchall()

        assert curve_rows == [("legacy_curve", 4.0, 0, 201)]
        assert scalar_rows == [
            ("execution", "batch_score", 1.5, None, None, 1001),
            ("run", "score", 1.0, 201, 1, 1001),
            ("run", "score", None, 202, 2, 1001),
        ]

    def test_run_scalars_require_matching_run_and_seed_ids(self, tmp_path: Path) -> None:
        """
        Reject scalar batches that cannot preserve the run-to-seed identity mapping.
        """
        with pytest.raises(ValueError, match="same length"):
            write_seed_batch_scalars(
                tmp_path / "metrics.sqlite",
                {"score": (1.0,)},
                experiment_id=10,
                execution_id=1001,
                run_ids=(201, 202),
                seed_ids=(1,),
            )

        with pytest.raises(ValueError, match="metric 'score' must have one value per run"):
            write_seed_batch_scalars(
                tmp_path / "metrics.sqlite",
                {"score": (1.0,)},
                experiment_id=10,
                execution_id=1001,
                run_ids=(201, 202),
                seed_ids=(1, 2),
            )


class TestSubsampleFactorValidation:
    def test_invalid_factor_raises(self, tmp_path: Path) -> None:
        """A subsample factor below one is rejected before anything is written."""
        with pytest.raises(ValueError, match="subsample factor must be >= 1"):
            write_seed_batch_curve(
                tmp_path / "m.db",
                [[1.0]],
                metric_name="r",
                experiment_id=1,
                execution_id=1,
                run_ids=[1],
                seed_ids=[0],
                subsample_factor=0,
            )
