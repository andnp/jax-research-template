"""SQLite storage backend for research-instrument.

Persists MetricFrames to a SQLite database so that metrics survive
process exit and can be queried after training completes.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from research_instrument.collector import MetricFrame

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    global_step INTEGER NOT NULL,
    seed_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (name);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics (global_step);
CREATE INDEX IF NOT EXISTS idx_metrics_name_seed ON metrics (name, seed_id);
"""


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

    def __init__(self, db_path: Path | str, *, batch_size: int = 100) -> None:
        self._db_path = Path(db_path)
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._buffer: list[MetricFrame] = []
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

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
            "INSERT INTO metrics (name, value, global_step, seed_id) VALUES (?, ?, ?, ?)",
            [(f.name, f.value, f.global_step, f.seed_id) for f in self._buffer],
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
                "SELECT name, value, global_step, seed_id FROM metrics WHERE name = ? AND seed_id = ? ORDER BY global_step",
                (name, seed_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT name, value, global_step, seed_id FROM metrics WHERE name = ? ORDER BY global_step",
                (name,),
            ).fetchall()

        return [MetricFrame(name=r[0], value=r[1], global_step=r[2], seed_id=r[3]) for r in rows]

    def metric_names(self) -> list[str]:
        """Return all distinct metric names stored in the database."""
        with self._lock:
            self._flush_locked()
        rows = self._conn.execute("SELECT DISTINCT name FROM metrics ORDER BY name").fetchall()
        return [r[0] for r in rows]
