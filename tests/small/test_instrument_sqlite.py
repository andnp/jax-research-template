"""Small tests for research_instrument.sqlite_backend — persistence layer."""

import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from research_instrument.collector import MetricFrame
from research_instrument.sqlite_backend import SQLiteBackend


def make_backend(db_path: Path, *, batch_size: int = 100) -> SQLiteBackend:
    return SQLiteBackend(
        db_path,
        experiment_id=10,
        run_id=101,
        execution_id=1001,
        batch_size=batch_size,
    )


class TestSQLiteBackendBasic:
    def test_requires_explicit_experiment_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend_cls = cast(Callable[..., object], SQLiteBackend)
            with pytest.raises(TypeError):
                backend_cls(Path(tmpdir) / "test.db")

    def test_write_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = make_backend(Path(tmpdir) / "test.db", batch_size=1)
            frames = [MetricFrame(name="reward", value=1.5, global_step=0, seed_id=0)]
            backend.write_batch(frames)
            result = backend.query("reward")
            assert len(result) == 1
            assert result[0].value == 1.5
            assert result[0].global_step == 0
            backend.close()

    def test_batch_buffering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = make_backend(Path(tmpdir) / "test.db", batch_size=5)
            for i in range(3):
                backend.write_batch([MetricFrame(name="loss", value=float(i), global_step=i, seed_id=0)])
            # Not yet flushed (batch_size=5, only 3 items)
            # But query should auto-flush
            result = backend.query("loss")
            assert len(result) == 3
            backend.close()

    def test_flush_on_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = make_backend(db_path)
            backend.write_batch([MetricFrame(name="x", value=42.0, global_step=0, seed_id=0)])
            backend.close()

            # Reopen and verify data persisted
            backend2 = make_backend(db_path)
            result = backend2.query("x")
            assert len(result) == 1
            assert result[0].value == 42.0
            backend2.close()

    def test_query_with_seed_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = make_backend(Path(tmpdir) / "test.db", batch_size=1)
            backend.write_batch([
                MetricFrame(name="reward", value=1.0, global_step=0, seed_id=0),
                MetricFrame(name="reward", value=2.0, global_step=0, seed_id=1),
            ])
            result = backend.query("reward", seed_id=1)
            assert len(result) == 1
            assert result[0].value == 2.0
            backend.close()

    def test_metric_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = make_backend(Path(tmpdir) / "test.db", batch_size=1)
            backend.write_batch([
                MetricFrame(name="reward", value=1.0, global_step=0, seed_id=0),
                MetricFrame(name="loss", value=0.5, global_step=0, seed_id=0),
                MetricFrame(name="reward", value=2.0, global_step=1, seed_id=0),
            ])
            names = backend.metric_names()
            assert sorted(names) == ["loss", "reward"]
            backend.close()

    def test_ordering_by_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = make_backend(Path(tmpdir) / "test.db", batch_size=1)
            backend.write_batch([
                MetricFrame(name="r", value=3.0, global_step=2, seed_id=0),
                MetricFrame(name="r", value=1.0, global_step=0, seed_id=0),
                MetricFrame(name="r", value=2.0, global_step=1, seed_id=0),
            ])
            result = backend.query("r")
            steps = [f.global_step for f in result]
            assert steps == [0, 1, 2]
            backend.close()

    def test_empty_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = make_backend(Path(tmpdir) / "test.db")
            result = backend.query("nonexistent")
            assert result == []
            backend.close()

    def test_identity_bound_backends_can_share_one_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "shared.db"
            backend_one = SQLiteBackend(
                db_path,
                batch_size=1,
                experiment_id=10,
                run_id=101,
                execution_id=1001,
            )
            backend_two = SQLiteBackend(
                db_path,
                batch_size=1,
                experiment_id=10,
                run_id=102,
                execution_id=1002,
            )

            backend_one.write_batch([MetricFrame(name="reward", value=1.0, global_step=5, seed_id=0)])
            backend_two.write_batch([MetricFrame(name="reward", value=2.0, global_step=5, seed_id=0)])
            backend_one.close()
            backend_two.close()

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT experiment_id, run_id, execution_id, metric_name, global_step, value "
                    "FROM metrics ORDER BY execution_id"
                ).fetchall()
                assert rows == [
                    (10, 101, 1001, "reward", 5, 1.0),
                    (10, 102, 1002, "reward", 5, 2.0),
                ]

                index_names = {
                    row[1]
                    for row in conn.execute("PRAGMA index_list(metrics)").fetchall()
                }
                assert "idx_metrics_experiment_metric_name_step" in index_names
                assert "idx_metrics_run_metric_name_step" in index_names
                assert "idx_metrics_execution_metric_name_step" in index_names

    def test_existing_table_is_migrated_to_metric_name_when_identity_is_present(self):
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

            backend = make_backend(db_path, batch_size=1)
            backend.write_batch([MetricFrame(name="reward", value=3.0, global_step=7, seed_id=2)])
            result = backend.query("reward", seed_id=2)
            backend.close()

            assert len(result) == 2
            assert [frame.value for frame in result] == [2.5, 3.0]

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
                    "FROM metrics ORDER BY global_step"
                ).fetchall()
                assert stored_rows == [
                    (10, 101, 1001, "reward", 6, 2.5),
                    (10, 101, 1001, "reward", 7, 3.0),
                ]

    def test_legacy_table_without_identity_columns_is_rejected(self):
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
                make_backend(db_path)
