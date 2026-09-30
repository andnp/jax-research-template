"""Connection-scoped persistence operations for physical executions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from .parameter import ParameterValue
from .records import ExecutionArtifactRow, ExecutionRow, _record_from_row
from .status import LIVE, TRANSITIONS, ExecutionStatus

# The planned batch's identity; the dead-worker sweep and resume read these back.
_PLANNER_KEYS = frozenset({"run_ids", "static_config_json", "vmap_zone_json"})


def add_execution(
    conn: sqlite3.Connection,
    hostname: str | None,
    git_commit: str | None,
    git_diff_blob: str | None,
    jax_config: Mapping[str, ParameterValue] | None,
) -> int:
    return _insert_execution(conn, hostname, git_commit, git_diff_blob, jax_config)


def _insert_execution(
    conn: sqlite3.Connection,
    hostname: str | None,
    git_commit: str | None,
    git_diff_blob: str | None,
    jax_config: Mapping[str, ParameterValue] | None,
) -> int:
    jax_json = json.dumps(jax_config, sort_keys=True) if jax_config is not None else None
    cur = conn.execute(
        "INSERT INTO Executions(hostname, git_commit, git_diff_blob, jax_config_json) "
        "VALUES (?, ?, ?, ?)",
        (hostname, git_commit, git_diff_blob, jax_json),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Insert failed: no execution ID returned.")
    return int(cur.lastrowid)


def record_execution_git_metadata(
    conn: sqlite3.Connection,
    execution_id: int,
    git_commit: str | None,
    git_diff_blob: str | None,
) -> None:
    conn.execute(
        "UPDATE Executions SET git_commit = COALESCE(?, git_commit), "
        "git_diff_blob = COALESCE(?, git_diff_blob) WHERE id = ?",
        (git_commit, git_diff_blob, execution_id),
    )


def get_execution(conn: sqlite3.Connection, execution_id: int) -> ExecutionRow | None:
    row = conn.execute(
        "SELECT id, status, hostname, start_time, end_time, git_commit, git_diff_blob, jax_config_json "
        "FROM Executions WHERE id = ?",
        (execution_id,),
    ).fetchone()
    return _record_from_row(ExecutionRow, row) if row else None


def record_execution_artifacts(
    conn: sqlite3.Connection,
    execution_id: int,
    root_path: str,
    manifest_path: str | None,
    metadata: dict[str, object] | None,
) -> None:
    _record_execution_artifacts(conn, execution_id, root_path, manifest_path, metadata)


def _record_execution_artifacts(
    conn: sqlite3.Connection,
    execution_id: int,
    root_path: str,
    manifest_path: str | None,
    metadata: dict[str, object] | None,
) -> None:
    existing = conn.execute(
        "SELECT metadata_json FROM ExecutionArtifacts WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    merged_metadata: dict[str, object] = {}
    if existing is not None and existing[0] is not None:
        parsed = json.loads(existing[0])
        if not isinstance(parsed, dict):
            raise ValueError("Execution artifact metadata must be a JSON object.")
        merged_metadata.update(parsed)
    if metadata is not None:
        clobbered = sorted(
            key for key in _PLANNER_KEYS & metadata.keys() & merged_metadata.keys() if metadata[key] != merged_metadata[key]
        )
        if clobbered:
            raise ValueError(f"Execution {execution_id} metadata must not overwrite planner keys {clobbered}.")
        merged_metadata.update(metadata)
    metadata_json = json.dumps(merged_metadata, sort_keys=True) if merged_metadata else None
    conn.execute(
        "INSERT INTO ExecutionArtifacts(execution_id, root_path, manifest_path, metadata_json) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(execution_id) DO UPDATE SET "
        "root_path = excluded.root_path, "
        "manifest_path = excluded.manifest_path, "
        "metadata_json = excluded.metadata_json",
        (execution_id, root_path, manifest_path, metadata_json),
    )


def get_execution_artifacts(conn: sqlite3.Connection, execution_id: int) -> ExecutionArtifactRow | None:
    row = conn.execute(
        "SELECT execution_id, root_path, manifest_path, metadata_json, created_at "
        "FROM ExecutionArtifacts WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    return _record_from_row(ExecutionArtifactRow, row) if row else None


def update_execution_status(
    conn: sqlite3.Connection,
    execution_id: int,
    status: str,
    start_time: str | None,
    end_time: str | None,
) -> None:
    if status not in TRANSITIONS:
        raise ValueError(f"Invalid status {status!r}.")
    conn.execute("BEGIN IMMEDIATE")
    try:
        _transition(conn, execution_id, status, start_time, end_time)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def fail_live_execution(conn: sqlite3.Connection, execution_id: int, end_time: str) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT status FROM Executions WHERE id = ?", (execution_id,)).fetchone()
        if row is not None and row[0] in LIVE:
            _transition(conn, execution_id, ExecutionStatus.FAILED, None, end_time)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _transition(
    conn: sqlite3.Connection,
    execution_id: int,
    status: str,
    start_time: str | None,
    end_time: str | None,
) -> None:
    row = conn.execute("SELECT status FROM Executions WHERE id = ?", (execution_id,)).fetchone()
    if row is None:
        raise ValueError(f"Execution {execution_id} does not exist.")
    current_status = ExecutionStatus(row[0])
    if status not in TRANSITIONS[current_status]:
        raise ValueError(f"Illegal execution transition: {current_status} -> {status}.")
    updated = conn.execute(
        "UPDATE Executions SET status = ?, start_time = COALESCE(?, start_time), "
        "end_time = COALESCE(?, end_time) WHERE id = ? AND status = ?",
        (status, start_time, end_time, execution_id, current_status),
    )
    if updated.rowcount != 1:
        raise RuntimeError(f"Execution {execution_id} changed before it could be updated.")


def reclaim_stale_executions(
    conn: sqlite3.Connection,
    experiment_id: int,
    stale_after_seconds: float,
) -> list[int]:
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            "SELECT DISTINCT e.id FROM Executions e "
            "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
            "INNER JOIN Runs r ON r.id = er.run_id "
            f"WHERE r.experiment_id = ? AND e.status = '{ExecutionStatus.RUNNING}' "
            "AND e.start_time IS NOT NULL AND e.start_time < ?",
            (experiment_id, cutoff),
        ).fetchall()
        reclaimed = [int(row[0]) for row in rows]
        for execution_id in reclaimed:
            _transition(conn, execution_id, ExecutionStatus.FAILED, None, now)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return reclaimed


def link_execution_run(conn: sqlite3.Connection, execution_id: int, run_id: int) -> None:
    _link_execution_run(conn, execution_id, run_id)


def _link_execution_run(conn: sqlite3.Connection, execution_id: int, run_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO ExecutionRuns(execution_id, run_id) VALUES (?, ?)",
        (execution_id, run_id),
    )
