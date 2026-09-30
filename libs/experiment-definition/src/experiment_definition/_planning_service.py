"""Connection-scoped selection and transaction-neutral planning operations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import _execution_repository as _execution
from ._database_utils import _json_stable
from ._types import PlannedExecution, RunBatch
from .parameter import ParameterValue
from .records import RunRow, _record_from_row

_CLAIMED_RUNS_SQL = (
    "SELECT DISTINCT er.run_id AS run_id "
    "FROM ExecutionRuns er "
    "INNER JOIN Executions e ON e.id = er.execution_id "
    "WHERE e.status IN ('PENDING', 'RUNNING', 'COMPLETED')"
)


def _static_config_json(hyper_json: str, vmap_zone_json: str | None) -> str:
    hyper_params = json.loads(hyper_json)
    if vmap_zone_json is None:
        return _json_stable(hyper_params)

    vmap_zone = json.loads(vmap_zone_json)
    static_keys = list(vmap_zone.get("static_keys", []))
    if not static_keys:
        return _json_stable(hyper_params)

    static_params = {key: hyper_params[key] for key in static_keys if key in hyper_params}
    return _json_stable(static_params)


def _run_matches(hyperparams: Mapping[str, ParameterValue], run: RunRow, only: Mapping[str, str]) -> bool:
    fields: dict[str, object] = {**hyperparams, "seed": run.seed, "ablation": run.ablation}
    return all(
        key in fields and str(fields[key]) in set(value.split(","))
        for key, value in only.items()
    )


type _BatchKey = tuple[int | None, int | None, str | None, str, str | None]


def _batch_key(
    algo_version_id: int | None,
    env_version_id: int | None,
    ablation: str | None,
    hyper_json: str,
    vmap_zone_json: str | None,
) -> _BatchKey:
    static_config_json = _static_config_json(hyper_json, vmap_zone_json)
    return (algo_version_id, env_version_id, ablation, static_config_json, vmap_zone_json)


def _group_key(batch_key: _BatchKey) -> str:
    return hashlib.sha256(json.dumps(batch_key).encode()).hexdigest()[:12]


def list_unsatisfied_run_batches(
    conn: sqlite3.Connection,
    experiment_id: int,
    max_runs_per_batch: int | None,
    only: Mapping[str, str] | None,
    group: str | None,
) -> list[RunBatch]:
    rows = conn.execute(
        "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation, "
        "       hc.json_blob, hc.vmap_zone_json "
        "FROM Runs r "
        "INNER JOIN HyperparamConfigs hc ON hc.id = r.hyper_id "
        "LEFT JOIN (" + _CLAIMED_RUNS_SQL + ") claimed ON claimed.run_id = r.id "
        "WHERE r.experiment_id = ? AND claimed.run_id IS NULL "
        "ORDER BY r.hyper_id, r.seed, r.id",
        (experiment_id,),
    ).fetchall()

    grouped_runs: dict[_BatchKey, list[int]] = {}
    for row in rows:
        run = _record_from_row(RunRow, row)
        hyper_json = str(row[7])
        if only and not _run_matches(json.loads(hyper_json), run, only):
            continue
        vmap_zone_json = str(row[8]) if row[8] is not None else None
        batch_key = _batch_key(run.algo_version_id, run.env_version_id, run.ablation, hyper_json, vmap_zone_json)
        grouped_runs.setdefault(batch_key, []).append(run.id)

    # None sorts first, so groups with and without an ablation or version still compare.
    def batch_key_of(
        item: tuple[_BatchKey, list[int]],
    ) -> tuple[tuple[bool, int | str], ...]:
        return tuple((value is not None, value if value is not None else 0) for value in item[0])

    batches: list[RunBatch] = []
    for batch_key, run_ids in sorted(grouped_runs.items(), key=batch_key_of):
        algo_version_id, env_version_id, ablation, static_config_json, vmap_zone_json = batch_key
        group_key = _group_key(batch_key)
        if group is not None and group_key != group:
            continue
        static_config = json.loads(static_config_json)
        vmap_zone = json.loads(vmap_zone_json) if vmap_zone_json is not None else None
        chunk_size = len(run_ids) if max_runs_per_batch is None else max_runs_per_batch
        for start in range(0, len(run_ids), chunk_size):
            batches.append(
                RunBatch(
                    algo_version_id,
                    env_version_id,
                    ablation,
                    static_config,
                    vmap_zone,
                    run_ids[start : start + chunk_size],
                    group_key,
                )
            )
    return batches


def fail_group_executions_started_since(
    conn: sqlite3.Connection,
    experiment_id: int,
    group: str,
    since: str,
) -> list[int]:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            "SELECT DISTINCT e.id, r.algo_version_id, r.env_version_id, r.ablation, hc.json_blob, hc.vmap_zone_json "
            "FROM Executions e "
            "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
            "INNER JOIN Runs r ON r.id = er.run_id "
            "INNER JOIN HyperparamConfigs hc ON hc.id = r.hyper_id "
            "WHERE r.experiment_id = ? AND e.status = 'RUNNING' AND e.start_time >= ?",
            (experiment_id, since),
        ).fetchall()
        failed = sorted(
            {
                int(row[0])
                for row in rows
                if _group_key(_batch_key(row[1], row[2], row[3], str(row[4]), row[5])) == group
            }
        )
        conn.executemany(
            "UPDATE Executions SET status = 'FAILED', end_time = ? WHERE id = ? AND status = 'RUNNING'",
            [(now, execution_id) for execution_id in failed],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return failed


def unsatisfied_run_ids(
    conn: sqlite3.Connection,
    experiment_id: int,
    limit: int | None,
) -> list[int] | None:
    rows = conn.execute(
        "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation "
        "FROM Runs r "
        "LEFT JOIN (" + _CLAIMED_RUNS_SQL + ") claimed ON claimed.run_id = r.id "
        "WHERE r.experiment_id = ? AND claimed.run_id IS NULL "
        "ORDER BY r.seed, r.hyper_id, r.id",
        (experiment_id,),
    ).fetchall()
    if not rows:
        return None

    unsatisfied_runs = [_record_from_row(RunRow, row) for row in rows]
    selected_runs = unsatisfied_runs if limit is None else unsatisfied_runs[:limit]
    return [run.id for run in selected_runs]


def _plan_execution(
    conn: sqlite3.Connection,
    run_ids: list[int],
    hostname: str | None,
    git_commit: str | None,
    git_diff_blob: str | None,
    jax_config: Mapping[str, ParameterValue] | None,
) -> int:
    execution_id = _execution._insert_execution(conn, hostname, git_commit, git_diff_blob, jax_config)
    for run_id in run_ids:
        _execution._link_execution_run(conn, execution_id, run_id)
    return execution_id


def plan_batch_with_artifacts(
    conn: sqlite3.Connection,
    batch: RunBatch,
    artifacts_root: str | Path,
    manifest_name: str,
    hostname: str | None,
    git_commit: str | None,
    git_diff_blob: str | None,
    jax_config: Mapping[str, ParameterValue] | None,
) -> PlannedExecution:
    execution_id = _plan_execution(conn, batch.run_ids, hostname, git_commit, git_diff_blob, jax_config)
    root_path = str(Path(artifacts_root) / str(execution_id))
    manifest_path = str(Path(root_path) / manifest_name)
    _execution._record_execution_artifacts(
        conn,
        execution_id,
        root_path,
        manifest_path,
        {
            "run_ids": batch.run_ids,
            "static_config_json": _json_stable(batch.static_config),
            "vmap_zone_json": _json_stable(batch.vmap_zone) if batch.vmap_zone is not None else None,
        },
    )
    return PlannedExecution(execution_id, batch.run_ids, root_path, manifest_path, batch.static_config, batch.vmap_zone)
