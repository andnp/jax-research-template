"""Schema and connection handling for metrics.sqlite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# SQLite stores NaN as NULL, so a nullable value column is what lets diverged runs be recorded.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    value REAL,
    global_step INTEGER NOT NULL,
    seed_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    execution_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_SCALAR_METRICS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scalar_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    value REAL,
    scope TEXT NOT NULL CHECK(scope IN ('run', 'execution')),
    experiment_id INTEGER NOT NULL,
    run_id INTEGER,
    seed_id INTEGER,
    execution_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (scope = 'run' AND run_id IS NOT NULL AND seed_id IS NOT NULL)
        OR (scope = 'execution' AND run_id IS NULL AND seed_id IS NULL)
    )
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

_CREATE_SCALAR_METRICS_INDEXES_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_scalar_metrics_run_identity
    ON scalar_metrics (experiment_id, execution_id, metric_name, run_id)
    WHERE scope = 'run';
CREATE UNIQUE INDEX IF NOT EXISTS idx_scalar_metrics_execution_identity
    ON scalar_metrics (experiment_id, execution_id, metric_name)
    WHERE scope = 'execution';
CREATE INDEX IF NOT EXISTS idx_scalar_metrics_experiment_metric
    ON scalar_metrics (experiment_id, metric_name, execution_id);
CREATE INDEX IF NOT EXISTS idx_scalar_metrics_run_metric
    ON scalar_metrics (run_id, metric_name, execution_id)
    WHERE run_id IS NOT NULL;
"""

METRICS_COLUMNS = (
    "metric_name",
    "value",
    "global_step",
    "seed_id",
    "experiment_id",
    "run_id",
    "execution_id",
)
SCALAR_METRICS_COLUMNS = (
    "metric_name",
    "value",
    "scope",
    "experiment_id",
    "run_id",
    "seed_id",
    "execution_id",
)


def _insert_sql(table: str, columns: tuple[str, ...]) -> str:
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


INSERT_METRICS_SQL = _insert_sql("metrics", METRICS_COLUMNS)
INSERT_SCALAR_METRICS_SQL = _insert_sql("scalar_metrics", SCALAR_METRICS_COLUMNS)

_STRICT_COLUMN_NAMES = {"id", "created_at", *METRICS_COLUMNS}


def open_metrics_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open a metrics.sqlite connection with a busy timeout and the schema ready.

    metrics.sqlite is written concurrently by multiple Slurm array tasks that
    may run on different nodes, sharing the file over a network filesystem
    (NFS/Lustre). WAL requires its wal-index to live in shared memory backed
    by mmap on a single host, and SQLite's own docs say WAL does not work over
    a network filesystem — it can silently corrupt the database there. So we
    keep the default rollback journal and rely on busy_timeout to make a
    writer that hits contention wait and retry instead of failing immediately
    with "database is locked".
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=30000")
    _ensure_rollback_journal(conn, db_path)
    _initialize_schema(conn)
    return conn


def _ensure_rollback_journal(conn: sqlite3.Connection, db_path: Path | str) -> None:
    """Convert a database left in WAL mode by an earlier version back to rollback.

    Journal mode is stored in the database file itself, not chosen per connection, so a
    metrics.sqlite created while WAL was enabled stays WAL — and unsafe over NFS/Lustre —
    until something converts it.
    """
    mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    if mode != "wal":
        return
    new_mode = str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
    if new_mode == "wal":
        raise RuntimeError(
            f"{db_path} is stuck in WAL journal mode, likely because another connection "
            "still holds it open; close it and retry"
        )


def _initialize_schema(conn: sqlite3.Connection) -> None:
    """Create or upgrade the metrics schema in place."""
    existing_columns = _metrics_table_info(conn)
    if not existing_columns:
        conn.execute(_CREATE_TABLE_SQL)
    elif not _has_strict_metrics_schema(existing_columns):
        _migrate_metrics_table(conn, existing_columns)

    conn.execute(_CREATE_SCALAR_METRICS_TABLE_SQL)
    conn.executescript(_CREATE_INDEXES_SQL)
    conn.executescript(_CREATE_SCALAR_METRICS_INDEXES_SQL)
    conn.commit()


def _metrics_table_info(conn: sqlite3.Connection) -> dict[str, tuple[str, bool]]:
    return {
        str(row[1]): (str(row[2]), bool(row[3])) for row in conn.execute("PRAGMA table_info(metrics)").fetchall()
    }


def _has_strict_metrics_schema(columns: dict[str, tuple[str, bool]]) -> bool:
    return (
        columns.keys() == _STRICT_COLUMN_NAMES
        and not columns["value"][1]
        and all(
            columns[column_name][1]
            for column_name in ("metric_name", "global_step", "seed_id", "experiment_id", "run_id", "execution_id")
        )
    )


def _migrate_metrics_table(conn: sqlite3.Connection, columns: dict[str, tuple[str, bool]]) -> None:
    """Upgrade a legacy metrics table in place, preserving existing rows.

    Kept so a database created before NaN support (``value REAL NOT NULL``,
    or the older ``name`` column) migrates forward instead of failing outright.
    """
    source_metric_column = "metric_name" if "metric_name" in columns else "name" if "name" in columns else None
    if source_metric_column is None:
        raise RuntimeError("metrics table is missing the canonical metric name column")

    missing_identity_columns = [
        column_name for column_name in ("experiment_id", "run_id", "execution_id") if column_name not in columns
    ]
    if missing_identity_columns:
        joined_columns = ", ".join(missing_identity_columns)
        raise RuntimeError(f"metrics table uses a legacy schema without required identity columns: {joined_columns}")

    missing_identity_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE experiment_id IS NULL OR run_id IS NULL OR execution_id IS NULL"
        ).fetchone()[0]
    )
    if missing_identity_count:
        raise RuntimeError(
            "metrics table contains legacy rows without full experiment identity; create a fresh database or migrate those rows explicitly"
        )

    conn.execute("DROP TABLE IF EXISTS metrics__new")
    conn.execute(
        "CREATE TABLE metrics__new ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "metric_name TEXT NOT NULL, "
        "value REAL, "
        "global_step INTEGER NOT NULL, "
        "seed_id INTEGER NOT NULL, "
        "experiment_id INTEGER NOT NULL, "
        "run_id INTEGER NOT NULL, "
        "execution_id INTEGER NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    conn.execute(
        "INSERT INTO metrics__new ("
        "id, metric_name, value, global_step, seed_id, experiment_id, run_id, execution_id, created_at"
        ") SELECT "
        f"id, {source_metric_column}, value, global_step, seed_id, experiment_id, run_id, execution_id, created_at "
        "FROM metrics"
    )
    conn.execute("DROP TABLE metrics")
    conn.execute("ALTER TABLE metrics__new RENAME TO metrics")
