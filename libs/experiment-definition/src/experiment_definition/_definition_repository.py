"""Connection-scoped persistence operations for experiment definitions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ._database_utils import _hash_dict, _json_stable
from .parameter import ParameterValue
from .records import (
    ComponentRow,
    ComponentVersionProvenanceRow,
    ComponentVersionRow,
    ExperimentRow,
    HyperparamConfigRow,
    RunRow,
    _record_from_row,
)

if TYPE_CHECKING:
    from .experiment import _ExperimentState


def _upsert_component(cur: sqlite3.Cursor, name: str, comp_type: str, code_hash: str) -> tuple[int, int]:
    cur.execute("INSERT OR IGNORE INTO Components(name, type) VALUES (?, ?)", (name, comp_type))
    cur.execute("SELECT id FROM Components WHERE name = ?", (name,))
    row = cur.fetchone()
    assert row is not None
    comp_id: int = row[0]

    cur.execute(
        "SELECT id, version_number, code_snapshot_hash FROM ComponentVersions "
        "WHERE component_id = ? ORDER BY version_number DESC LIMIT 1",
        (comp_id,),
    )
    existing = cur.fetchone()

    if existing is None or existing[2] != code_hash:
        next_version = 1 if existing is None else existing[1] + 1
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "INSERT INTO ComponentVersions(component_id, version_number, created_at, code_snapshot_hash) "
            "VALUES (?, ?, ?, ?)",
            (comp_id, next_version, now, code_hash),
        )
        assert cur.lastrowid is not None
        version_id: int = cur.lastrowid
    else:
        version_id = existing[0]

    return comp_id, version_id


def _upsert_hyperparam_config(
    cur: sqlite3.Cursor,
    config: dict[str, ParameterValue],
    vmap_zone: dict[str, list[str]] | None = None,
) -> int:
    h = _hash_dict(config)
    blob = _json_stable(config)
    vmap_json = _json_stable(vmap_zone) if vmap_zone is not None else None
    cur.execute(
        "INSERT OR IGNORE INTO HyperparamConfigs(hash, json_blob, vmap_zone_json) VALUES (?, ?, ?)",
        (h, blob, vmap_json),
    )
    cur.execute("SELECT id, vmap_zone_json FROM HyperparamConfigs WHERE hash = ?", (h,))
    row = cur.fetchone()
    assert row is not None
    result = int(row[0])
    existing_vmap_json = row[1]
    if vmap_json is not None and existing_vmap_json != vmap_json:
        cur.execute(
            "UPDATE HyperparamConfigs SET vmap_zone_json = ? WHERE id = ?",
            (vmap_json, result),
        )
    return result


def _upsert_experiment(cur: sqlite3.Cursor, name: str, description: str | None) -> int:
    cur.execute("SELECT id FROM Experiments WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is not None:
        return int(row[0])

    cur.execute(
        "INSERT INTO Experiments(name, description) VALUES (?, ?)",
        (name, description),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Insert failed: no experiment ID returned.")
    return int(cur.lastrowid)


def sync_runs(
    cur: sqlite3.Cursor,
    exp_id: int,
    comp_version_ids: dict[str, int],
    configs: list[dict[str, ParameterValue]],
    state: "_ExperimentState",
    *,
    static_parameter_names: frozenset[str] | None = None,
) -> set[int]:
    from .component import ComponentType

    algo_components = [comp for comp in state.components if comp.type == ComponentType.ALGO]
    env_components = [comp for comp in state.components if comp.type == ComponentType.ENV]
    algo_vid: int | None = comp_version_ids.get(algo_components[0].name) if len(algo_components) == 1 else None
    env_vid: int | None = comp_version_ids.get(env_components[0].name) if len(env_components) == 1 else None

    ablation_list: list[tuple[str | None, dict[str, ParameterValue]]] = [(None, {})]
    for abl in state.ablations:
        ablation_list.append((abl.name, abl.overrides))

    wanted: set[int] = set()
    for config in configs:
        raw_seed = config.get("seed", 0)
        if raw_seed is None:
            raise TypeError("experiment seed must not be None")
        seed = int(raw_seed)
        hyper_config = {k: v for k, v in config.items() if k != "seed"}

        for abl_name, abl_overrides in ablation_list:
            final_hyper = {**hyper_config, **abl_overrides}
            hyper_id = _upsert_hyperparam_config(
                cur,
                final_hyper,
                _vmap_zone_for_hyper_config(
                    state,
                    final_hyper,
                    static_parameter_names=static_parameter_names,
                ),
            )

            existing = cur.execute(
                "SELECT id FROM Runs "
                "WHERE experiment_id = ? AND algo_version_id IS ? AND env_version_id IS ? "
                "AND hyper_id = ? AND seed = ? AND ablation IS ?",
                (exp_id, algo_vid, env_vid, hyper_id, seed, abl_name),
            ).fetchone()
            if existing is not None:
                wanted.add(int(existing[0]))
                continue

            cur.execute(
                "INSERT INTO Runs(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exp_id, algo_vid, env_vid, hyper_id, seed, abl_name),
            )
            new_run_id = cur.lastrowid
            assert new_run_id is not None, "INSERT into Runs did not assign a rowid"
            wanted.add(new_run_id)
    return wanted


def _vmap_zone_for_hyper_config(
    state: "_ExperimentState",
    hyper_config: dict[str, ParameterValue],
    *,
    static_parameter_names: frozenset[str] | None = None,
) -> dict[str, list[str]]:
    static_keys = sorted(
        {
            param.name
            for param in state.parameters
            if (param.is_static if static_parameter_names is None else param.name in static_parameter_names)
            and param.name in hyper_config
        }
    )
    dynamic_keys = sorted(key for key in hyper_config if key not in static_keys)
    return {"static_keys": static_keys, "dynamic_keys": dynamic_keys}


def add_component(conn: sqlite3.Connection, name: str, component_type: str) -> int:
    cur = conn.execute(
        "INSERT INTO Components(name, type) VALUES (?, ?)",
        (name, component_type),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Insert failed: no component ID returned.")
    return int(cur.lastrowid)


def get_component(conn: sqlite3.Connection, name: str) -> ComponentRow | None:
    row = conn.execute("SELECT id, name, type FROM Components WHERE name = ?", (name,)).fetchone()
    return _record_from_row(ComponentRow, row) if row else None


def add_component_version(
    conn: sqlite3.Connection,
    component_id: int,
    code_snapshot_hash: str,
    notes: str | None,
) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version_number), 0) FROM ComponentVersions WHERE component_id = ?",
        (component_id,),
    ).fetchone()
    next_version = int(row[0]) + 1
    cur = conn.execute(
        "INSERT INTO ComponentVersions(component_id, version_number, code_snapshot_hash, notes) "
        "VALUES (?, ?, ?, ?)",
        (component_id, next_version, code_snapshot_hash, notes),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Insert failed: no component version ID returned.")
    return int(cur.lastrowid)


def get_latest_version(conn: sqlite3.Connection, component_id: int) -> ComponentVersionRow | None:
    row = conn.execute(
        "SELECT id, component_id, version_number, created_at, code_snapshot_hash, notes "
        "FROM ComponentLatestVersions WHERE component_id = ?",
        (component_id,),
    ).fetchone()
    return _record_from_row(ComponentVersionRow, row) if row else None


def get_component_version_provenance(
    conn: sqlite3.Connection,
    version_id: int,
) -> ComponentVersionProvenanceRow | None:
    row = conn.execute(
        "SELECT c.id AS component_id, c.name AS component_name, c.type AS component_type, "
        "cv.id AS version_id, cv.version_number, cv.code_snapshot_hash "
        "FROM ComponentVersions cv INNER JOIN Components c ON c.id = cv.component_id "
        "WHERE cv.id = ?",
        (version_id,),
    ).fetchone()
    return _record_from_row(ComponentVersionProvenanceRow, row) if row else None


def add_hyperparam_config(
    conn: sqlite3.Connection,
    params: Mapping[str, ParameterValue],
    vmap_zone: dict[str, list[str]] | None,
) -> int:
    json_blob = json.dumps(params, sort_keys=True)
    config_hash = _hash_dict(params)
    vmap_json = json.dumps(vmap_zone, sort_keys=True) if vmap_zone is not None else None

    conn.execute(
        "INSERT OR IGNORE INTO HyperparamConfigs(hash, json_blob, vmap_zone_json) VALUES (?, ?, ?)",
        (config_hash, json_blob, vmap_json),
    )
    row = conn.execute("SELECT id FROM HyperparamConfigs WHERE hash = ?", (config_hash,)).fetchone()
    assert row is not None
    return int(row[0])


def get_hyperparam_config(conn: sqlite3.Connection, config_id: int) -> HyperparamConfigRow | None:
    row = conn.execute(
        "SELECT id, hash, json_blob, vmap_zone_json FROM HyperparamConfigs WHERE id = ?",
        (config_id,),
    ).fetchone()
    if row is None:
        return None
    row_id, config_hash, json_blob, vmap_zone_json = row
    return HyperparamConfigRow(
        id=row_id,
        hash=config_hash,
        hyperparameters=json.loads(json_blob),
        vmap_zone=json.loads(vmap_zone_json) if vmap_zone_json is not None else None,
    )


def add_experiment(conn: sqlite3.Connection, name: str, description: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO Experiments(name, description) VALUES (?, ?)",
        (name, description),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Insert failed: no experiment ID returned.")
    return int(cur.lastrowid)


def get_experiment(conn: sqlite3.Connection, name: str) -> ExperimentRow | None:
    row = conn.execute(
        "SELECT id, name, description, created_at FROM Experiments WHERE name = ?",
        (name,),
    ).fetchone()
    return _record_from_row(ExperimentRow, row) if row else None


def list_experiments(conn: sqlite3.Connection) -> list[ExperimentRow]:
    rows = conn.execute(
        "SELECT id, name, description, created_at FROM Experiments ORDER BY name",
    ).fetchall()
    return [_record_from_row(ExperimentRow, row) for row in rows]


def get_experiment_by_id(conn: sqlite3.Connection, experiment_id: int) -> ExperimentRow | None:
    row = conn.execute(
        "SELECT id, name, description, created_at FROM Experiments WHERE id = ?",
        (experiment_id,),
    ).fetchone()
    return _record_from_row(ExperimentRow, row) if row else None


def add_run(
    conn: sqlite3.Connection,
    experiment_id: int,
    algo_version_id: int | None,
    env_version_id: int | None,
    hyper_id: int,
    seed: int,
    ablation: str | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO Runs(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation),
    )
    if cur.lastrowid is None:
        raise RuntimeError("Insert failed: no run ID returned.")
    return int(cur.lastrowid)


def list_runs(conn: sqlite3.Connection, experiment_id: int) -> list[RunRow]:
    rows = conn.execute(
        "SELECT id, experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation "
        "FROM Runs WHERE experiment_id = ? ORDER BY seed, hyper_id, id",
        (experiment_id,),
    ).fetchall()
    return [_record_from_row(RunRow, row) for row in rows]


def list_runs_without_completed_execution(conn: sqlite3.Connection, experiment_id: int) -> list[RunRow]:
    rows = conn.execute(
        "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation "
        "FROM Runs r LEFT JOIN ("
        "    SELECT DISTINCT er.run_id AS run_id FROM ExecutionRuns er "
        "    INNER JOIN Executions e ON e.id = er.execution_id WHERE e.status = 'COMPLETED'"
        ") completed ON completed.run_id = r.id "
        "WHERE r.experiment_id = ? AND completed.run_id IS NULL "
        "ORDER BY r.seed, r.hyper_id, r.id",
        (experiment_id,),
    ).fetchall()
    return [_record_from_row(RunRow, row) for row in rows]


def list_orphan_runs(
    conn: sqlite3.Connection,
    experiment_id: int,
    wanted_run_ids: frozenset[int],
) -> list[RunRow]:
    rows = conn.execute(
        "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation "
        "FROM Runs r LEFT JOIN ExecutionRuns er ON er.run_id = r.id "
        "WHERE r.experiment_id = ? AND er.run_id IS NULL "
        "ORDER BY r.hyper_id, r.seed, r.id",
        (experiment_id,),
    ).fetchall()
    return [_record_from_row(RunRow, row) for row in rows if int(row["id"]) not in wanted_run_ids]


def delete_runs(conn: sqlite3.Connection, run_ids: list[int]) -> int:
    deleted = 0
    for run_id in run_ids:
        cursor = conn.execute(
            "DELETE FROM Runs WHERE id = ? "
            "AND NOT EXISTS (SELECT 1 FROM ExecutionRuns er WHERE er.run_id = Runs.id)",
            (run_id,),
        )
        deleted += cursor.rowcount
    return deleted
