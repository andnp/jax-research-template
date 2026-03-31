"""SQLite storage backend for research-instrument.

Persists MetricFrames to a SQLite database so that metrics survive
process exit and can be queried after training completes.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from research_instrument.collector import MetricFrame

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    global_step INTEGER NOT NULL,
    seed_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    execution_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON metrics (metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics (global_step);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_name_seed ON metrics (metric_name, seed_id);
CREATE INDEX IF NOT EXISTS idx_metrics_experiment_metric_name_step ON metrics (experiment_id, metric_name, global_step);
CREATE INDEX IF NOT EXISTS idx_metrics_run_metric_name_step ON metrics (run_id, metric_name, global_step);
CREATE INDEX IF NOT EXISTS idx_metrics_execution_metric_name_step ON metrics (execution_id, metric_name, global_step);
"""

_STRICT_COLUMN_NAMES = {
    "id",
    "metric_name",
    "value",
    "global_step",
    "seed_id",
    "experiment_id",
    "run_id",
    "execution_id",
    "created_at",
}


class SQLiteBackend:
    """Thread-safe SQLite storage backend for metrics.

    Writes MetricFrames to a SQLite database, batching inserts within
    a single transaction for performance. All operations are protected
    by a threading lock since ``jax.debug.callback`` may invoke the
    backend from different threads.

    Args:
        db_path: Path to the SQLite database file. Created if absent.
        batch_size: Number of frames to accumulate before flushing to
            disk. A larger batch size reduces I/O overhead. Default 100.

    Example::

        backend = SQLiteBackend("metrics.db")
        collector = Collector(frozenset({"reward"}), backend=backend)
        # ... run training ...
        backend.flush()
        backend.close()
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        experiment_id: int,
        run_id: int,
        execution_id: int,
        batch_size: int = 100,
    ) -> None:
        self._db_path = Path(db_path)
        self._batch_size = batch_size
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._execution_id = execution_id
        self._lock = threading.Lock()
        self._buffer: list[MetricFrame] = []
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create or upgrade the metrics schema in place."""
        existing_columns = self._metrics_table_info()
        if not existing_columns:
            self._conn.execute(_CREATE_TABLE_SQL)
        elif not self._has_strict_metrics_schema(existing_columns):
            self._migrate_metrics_table(existing_columns)

        self._conn.executescript(_CREATE_INDEXES_SQL)
        self._conn.commit()

    def _metrics_table_info(self) -> dict[str, tuple[str, bool]]:
        return {
            str(row[1]): (str(row[2]), bool(row[3]))
            for row in self._conn.execute("PRAGMA table_info(metrics)").fetchall()
        }

    def _has_strict_metrics_schema(self, columns: dict[str, tuple[str, bool]]) -> bool:
        return columns.keys() == _STRICT_COLUMN_NAMES and all(
            columns[column_name][1]
            for column_name in ("metric_name", "value", "global_step", "seed_id", "experiment_id", "run_id", "execution_id")
        )

    def _migrate_metrics_table(self, columns: dict[str, tuple[str, bool]]) -> None:
        source_metric_column = "metric_name" if "metric_name" in columns else "name" if "name" in columns else None
        if source_metric_column is None:
            raise RuntimeError("metrics table is missing the canonical metric name column")

        missing_identity_columns = [
            column_name
            for column_name in ("experiment_id", "run_id", "execution_id")
            if column_name not in columns
        ]
        if missing_identity_columns:
            joined_columns = ", ".join(missing_identity_columns)
            raise RuntimeError(
                f"metrics table uses a legacy schema without required identity columns: {joined_columns}"
            )

        missing_identity_count = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM metrics WHERE experiment_id IS NULL OR run_id IS NULL OR execution_id IS NULL"
            ).fetchone()[0]
        )
        if missing_identity_count:
            raise RuntimeError(
                "metrics table contains legacy rows without full experiment identity; create a fresh database or migrate those rows explicitly"
            )

        self._conn.execute("DROP TABLE IF EXISTS metrics__new")
        self._conn.execute(
            "CREATE TABLE metrics__new ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "metric_name TEXT NOT NULL, "
            "value REAL NOT NULL, "
            "global_step INTEGER NOT NULL, "
            "seed_id INTEGER NOT NULL, "
            "experiment_id INTEGER NOT NULL, "
            "run_id INTEGER NOT NULL, "
            "execution_id INTEGER NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.execute(
            "INSERT INTO metrics__new ("
            "id, metric_name, value, global_step, seed_id, experiment_id, run_id, execution_id, created_at"
            ") SELECT "
            f"id, {source_metric_column}, value, global_step, seed_id, experiment_id, run_id, execution_id, created_at "
            "FROM metrics"
        )
        self._conn.execute("DROP TABLE metrics")
        self._conn.execute("ALTER TABLE metrics__new RENAME TO metrics")

    def write_batch(self, frames: list[MetricFrame]) -> None:
        """Buffer frames and flush when batch_size is reached."""
        with self._lock:
            self._buffer.extend(frames)
            if len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """Flush all buffered frames to disk."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining data and close the database connection."""
        self.flush()
        self._conn.close()

    def _flush_locked(self) -> None:
        """Write buffered frames to SQLite. Must be called with lock held."""
        if not self._buffer:
            return
        self._conn.executemany(
            "INSERT INTO metrics ("
            "metric_name, value, global_step, seed_id, experiment_id, run_id, execution_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f.name,
                    f.value,
                    f.global_step,
                    f.seed_id,
                    self._experiment_id,
                    self._run_id,
                    self._execution_id,
                )
                for f in self._buffer
            ],
        )
        self._conn.commit()
        self._buffer.clear()

    # --- Query helpers ---

    def query(
        self,
        name: str,
        *,
        seed_id: int | None = None,
    ) -> list[MetricFrame]:
        """Retrieve stored frames for a given metric name.

        Args:
            name: Metric name to query.
            seed_id: Optional seed filter. When None, returns all seeds.

        Returns:
            List of :class:`MetricFrame` ordered by global_step.
        """
        with self._lock:
            self._flush_locked()

        if seed_id is not None:
            rows = self._conn.execute(
                "SELECT metric_name, value, global_step, seed_id FROM metrics WHERE metric_name = ? AND seed_id = ? ORDER BY global_step",
                (name, seed_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT metric_name, value, global_step, seed_id FROM metrics WHERE metric_name = ? ORDER BY global_step",
                (name,),
            ).fetchall()

        return [MetricFrame(name=r[0], value=r[1], global_step=r[2], seed_id=r[3]) for r in rows]

    def metric_names(self) -> list[str]:
        """Return all distinct metric names stored in the database."""
        with self._lock:
            self._flush_locked()
        rows = self._conn.execute(
            "SELECT DISTINCT metric_name FROM metrics ORDER BY metric_name"
        ).fetchall()
        return [r[0] for r in rows]
