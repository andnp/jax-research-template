"""Connection-scoped execution queries and invalidation operations."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping

from . import _definition_repository as _definition
from . import _execution_repository as _execution
from ._types import PlannedExecution
from .records import ExecutionRow, RunRow, _record_from_row, _run_matches
from .status import FINISHED, ExecutionStatus, sql_in


def list_execution_runs(conn: sqlite3.Connection, execution_id: int) -> list[RunRow]:
    rows = conn.execute(
        "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation, r.vmap_zone_json AS vmap_zone "
        "FROM Runs r "
        "INNER JOIN ExecutionRuns er ON er.run_id = r.id "
        "WHERE er.execution_id = ? "
        "ORDER BY r.seed, r.hyper_id, r.id",
        (execution_id,),
    ).fetchall()
    return [_record_from_row(RunRow, row) for row in rows]


def get_latest_completed_execution_for_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> ExecutionRow | None:
    row = conn.execute(
        "SELECT e.id, e.status, e.hostname, e.start_time, e.end_time, e.git_commit, e.git_diff_blob, e.jax_config_json "
        "FROM Executions e "
        "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
        f"WHERE er.run_id = ? AND e.status = '{ExecutionStatus.COMPLETED}' "
        "ORDER BY COALESCE(e.end_time, e.start_time, '') DESC, e.id DESC "
        "LIMIT 1",
        (run_id,),
    ).fetchone()
    return _record_from_row(ExecutionRow, row) if row else None


def list_run_execution_statuses(conn: sqlite3.Connection, experiment_id: int) -> dict[int, frozenset[str]]:
    rows = conn.execute(
        "SELECT r.id, e.status FROM Runs r "
        "LEFT JOIN ExecutionRuns er ON er.run_id = r.id "
        "LEFT JOIN Executions e ON e.id = er.execution_id "
        "WHERE r.experiment_id = ?",
        (experiment_id,),
    ).fetchall()
    statuses: dict[int, set[str]] = {}
    for run_id, status in rows:
        run_statuses = statuses.setdefault(int(run_id), set())
        if status is not None:
            run_statuses.add(str(status))
    return {run_id: frozenset(values) for run_id, values in statuses.items()}


def invalidate_execution(conn: sqlite3.Connection, execution_id: int) -> bool:
    existing = _execution.get_execution(conn, execution_id)
    if existing is None:
        return False
    _execution.update_execution_status(conn, execution_id, ExecutionStatus.INVALID, None, None)
    return True


def find_finished_execution_ids_by_commit(conn: sqlite3.Connection, git_commit: str) -> list[int]:
    rows = conn.execute(
        f"SELECT id FROM Executions WHERE git_commit = ? AND status IN {sql_in(FINISHED)} ORDER BY id",
        (git_commit,),
    ).fetchall()
    return [int(row[0]) for row in rows]


def find_executions_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> list[ExecutionRow]:
    rows = conn.execute(
        "SELECT DISTINCT e.id, e.status, e.hostname, e.start_time, e.end_time, "
        "e.git_commit, e.git_diff_blob, e.jax_config_json "
        "FROM Executions e "
        "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
        "INNER JOIN Runs r ON r.id = er.run_id "
        f"WHERE r.experiment_id = ? AND e.status IN {sql_in(FINISHED)} "
        "ORDER BY e.id",
        (experiment_id,),
    ).fetchall()
    return [_record_from_row(ExecutionRow, row) for row in rows]


def find_executions_by_hyperparams(
    conn: sqlite3.Connection,
    experiment_id: int,
    where: Mapping[str, str],
) -> list[ExecutionRow]:
    return [
        execution
        for execution in find_executions_for_experiment(conn, experiment_id)
        if (runs := list_execution_runs(conn, execution.id)) and all(_run_selected(conn, run, where) for run in runs)
    ]


def _run_selected(conn: sqlite3.Connection, run: RunRow, where: Mapping[str, str]) -> bool:
    config = _definition.get_hyperparam_config(conn, run.hyper_id)
    return config is not None and _run_matches(config.hyperparameters, run, where)


def list_executions(
    conn: sqlite3.Connection,
    experiment_id: int | None = None,
    *,
    status: str | None = None,
    git_commit: str | None = None,
) -> list[ExecutionRow]:
    clauses: list[str] = []
    params: list[object] = []

    if experiment_id is not None:
        clauses.append(
            "e.id IN ("
            "  SELECT er.execution_id FROM ExecutionRuns er"
            "  INNER JOIN Runs r ON r.id = er.run_id"
            "  WHERE r.experiment_id = ?"
            ")"
        )
        params.append(experiment_id)
    if status is not None:
        clauses.append("e.status = ?")
        params.append(status)
    if git_commit is not None:
        clauses.append("e.git_commit = ?")
        params.append(git_commit)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        "SELECT e.id, e.status, e.hostname, e.start_time, e.end_time, "
        "e.git_commit, e.git_diff_blob, e.jax_config_json "
        f"FROM Executions e{where} ORDER BY e.id DESC",
        params,
    ).fetchall()
    return [_record_from_row(ExecutionRow, row) for row in rows]


def get_planned_execution(conn: sqlite3.Connection, execution_id: int) -> PlannedExecution | None:
    execution = _execution.get_execution(conn, execution_id)
    if execution is None:
        return None
    artifacts = _execution.get_execution_artifacts(conn, execution_id)
    if artifacts is None:
        return None
    runs = list_execution_runs(conn, execution_id)
    run_ids = [r.id for r in runs]
    metadata = json.loads(artifacts.metadata_json) if artifacts.metadata_json else {}
    vmap_zone_json = metadata.get("vmap_zone_json")
    return PlannedExecution(
        execution_id=execution_id,
        run_ids=run_ids,
        root_path=artifacts.root_path,
        manifest_path=artifacts.manifest_path or "",
        static_config=json.loads(metadata.get("static_config_json", "{}")),
        vmap_zone=json.loads(vmap_zone_json) if vmap_zone_json is not None else None,
    )
