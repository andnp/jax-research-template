"""Small tests for research_instrument.sqlite_backend — persistence layer."""

import tempfile
from pathlib import Path

from research_instrument.collector import MetricFrame
from research_instrument.sqlite_backend import SQLiteBackend


class TestSQLiteBackendBasic:
    def test_write_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(Path(tmpdir) / "test.db", batch_size=1)
            frames = [MetricFrame(name="reward", value=1.5, global_step=0, seed_id=0)]
            backend.write_batch(frames)
            result = backend.query("reward")
            assert len(result) == 1
            assert result[0].value == 1.5
            assert result[0].global_step == 0
            backend.close()

    def test_batch_buffering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(Path(tmpdir) / "test.db", batch_size=5)
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
            backend = SQLiteBackend(db_path, batch_size=100)
            backend.write_batch([MetricFrame(name="x", value=42.0, global_step=0, seed_id=0)])
            backend.close()

            # Reopen and verify data persisted
            backend2 = SQLiteBackend(db_path)
            result = backend2.query("x")
            assert len(result) == 1
            assert result[0].value == 42.0
            backend2.close()

    def test_query_with_seed_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(Path(tmpdir) / "test.db", batch_size=1)
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
            backend = SQLiteBackend(Path(tmpdir) / "test.db", batch_size=1)
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
            backend = SQLiteBackend(Path(tmpdir) / "test.db", batch_size=1)
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
            backend = SQLiteBackend(Path(tmpdir) / "test.db")
            result = backend.query("nonexistent")
            assert result == []
            backend.close()
