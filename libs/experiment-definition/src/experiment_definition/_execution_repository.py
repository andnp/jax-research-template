"""Connection-scoped persistence operations for physical executions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping

from .parameter import ParameterValue
from .records import ExecutionArtifactRow, ExecutionRow, _record_from_row, execution_columns
from .status import LIVE, TRANSITIONS, ExecutionStatus, sql_in

# The planned batch's identity; the dead-worker sweep and resume read these back.
_PLANNER_KEYS = frozenset({"static_config_json", "vmap_zone_json"})


def add_execution(
    conn: sqlite3.Connection,
    hostname: str | None,
    git_commit: str | None,
    git_diff_blob: str | None,
    jax_config: Mapping[str, ParameterValue] | None,
    owner: str | None,
) -> int:
    jax_json = json.dumps(jax_config, sort_keys=True) if jax_config is not None else None
    cur = conn.execute(
        "INSERT INTO Executions(hostname, git_commit, git_diff_blob, jax_config_json, owner) "
        "VALUES (?, ?, ?, ?, ?)",
        (hostname, git_commit, git_diff_blob, jax_json, owner),
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
        f"SELECT {execution_columns()} FROM Executions WHERE id = ?",
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


def list_live_execution_owners(conn: sqlite3.Connection, experiment_id: int) -> dict[int, str | None]:
    rows = conn.execute(
        "SELECT DISTINCT e.id, e.owner FROM Executions e "
        "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
        "INNER JOIN Runs r ON r.id = er.run_id "
        f"WHERE r.experiment_id = ? AND e.status IN {sql_in(LIVE)} ORDER BY e.id",
        (experiment_id,),
    ).fetchall()
    return {int(row[0]): row[1] for row in rows}


def transfer_owners(conn: sqlite3.Connection, expected: Mapping[int, str], new: Mapping[int, str]) -> list[int]:
    """Give each live execution in ``expected`` the owner ``new`` names for it, if it still has the expected one."""
    transferred: list[int] = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        for execution_id, owner in expected.items():
            updated = conn.execute(
                f"UPDATE Executions SET owner = ? WHERE id = ? AND owner = ? AND status IN {sql_in(LIVE)}",
                (new[execution_id], execution_id, owner),
            )
            if updated.rowcount == 1:
                transferred.append(execution_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return transferred


def _transition_many(
    conn: sqlite3.Connection,
    execution_ids: Iterable[int],
    status: ExecutionStatus,
    end_time: str | None = None,
    expected_owners: Mapping[int, str] | None = None,
) -> list[int]:
    """Move every execution that may legally enter ``status`` there in one transaction, skipping the rest.

    With ``expected_owners``, an execution whose owner is no longer the expected one is skipped too.
    """
    transitioned: list[int] = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        for execution_id in execution_ids:
            row = conn.execute("SELECT status, owner FROM Executions WHERE id = ?", (execution_id,)).fetchone()
            if row is None or status not in TRANSITIONS[ExecutionStatus(row[0])]:
                continue
            if expected_owners is not None and row[1] != expected_owners.get(execution_id):
                continue
            _transition(conn, execution_id, status, None, end_time)
            transitioned.append(execution_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return transitioned


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


def link_execution_run(conn: sqlite3.Connection, execution_id: int, run_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO ExecutionRuns(execution_id, run_id) VALUES (?, ?)",
        (execution_id, run_id),
    )
