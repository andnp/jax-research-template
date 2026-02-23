"""SQLite persistence layer implementing the ADR 008 schema.

All DDL and upsert logic lives here so ``experiment.py`` stays free of SQL.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import _ExperimentState

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS Components (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE,
    type    TEXT NOT NULL DEFAULT 'OTHER'
);

CREATE TABLE IF NOT EXISTS ComponentVersions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id        INTEGER NOT NULL REFERENCES Components(id),
    version_number      INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL,
    code_snapshot_hash  TEXT NOT NULL,
    notes               TEXT,
    UNIQUE(component_id, version_number)
);

CREATE TABLE IF NOT EXISTS HyperparamConfigs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hash        TEXT NOT NULL UNIQUE,
    json_blob   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS Runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    INTEGER NOT NULL REFERENCES Experiments(id),
    algo_version_id  INTEGER REFERENCES ComponentVersions(id),
    env_version_id   INTEGER REFERENCES ComponentVersions(id),
    hyper_id         INTEGER NOT NULL REFERENCES HyperparamConfigs(id),
    seed             INTEGER NOT NULL,
    ablation         TEXT,
    UNIQUE(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation)
);

CREATE TABLE IF NOT EXISTS Executions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    hostname        TEXT,
    start_time      TEXT,
    end_time        TEXT,
    git_commit      TEXT,
    git_diff_blob   TEXT,
    jax_config_json TEXT
);

CREATE TABLE IF NOT EXISTS ExecutionRuns (
    execution_id    INTEGER NOT NULL REFERENCES Executions(id),
    run_id          INTEGER NOT NULL REFERENCES Runs(id),
    PRIMARY KEY (execution_id, run_id)
);

CREATE TABLE IF NOT EXISTS ParameterSpecs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES Experiments(id),
    name            TEXT NOT NULL,
    values_json     TEXT NOT NULL,
    is_static       INTEGER NOT NULL DEFAULT 0,
    component_id    INTEGER REFERENCES Components(id),
    conditions_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS MetricSpecs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES Experiments(id),
    name            TEXT NOT NULL,
    type            TEXT NOT NULL,
    frequency       TEXT NOT NULL,
    UNIQUE(experiment_id, name)
);
"""


def _json_stable(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _hash_dict(d: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_stable(d).encode()).hexdigest()


def _upsert_component(cur: sqlite3.Cursor, name: str, comp_type: str, code_hash: str) -> tuple[int, int]:
    """Insert or look up a component and ensure a version row exists.

    Returns:
        Tuple of (component_id, component_version_id).
    """
    cur.execute("INSERT OR IGNORE INTO Components(name, type) VALUES (?, ?)", (name, comp_type))
    cur.execute("SELECT id FROM Components WHERE name = ?", (name,))
    row = cur.fetchone()
    assert row is not None
    comp_id: int = row[0]

    # Find the latest version for this component
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


def _upsert_hyperparam_config(cur: sqlite3.Cursor, config: dict[str, object]) -> int:
    h = _hash_dict(config)
    blob = _json_stable(config)
    cur.execute("INSERT OR IGNORE INTO HyperparamConfigs(hash, json_blob) VALUES (?, ?)", (h, blob))
    cur.execute("SELECT id FROM HyperparamConfigs WHERE hash = ?", (h,))
    row = cur.fetchone()
    assert row is not None
    result: int = row[0]
    return result


def sync_to_db(db_path: Path | str, state: "_ExperimentState") -> None:
    """Materialise the experiment definition into a SQLite file.

    Creates the database if it does not exist and applies the ADR 008 schema.
    All operations are wrapped in a single transaction.

    Args:
        db_path: Path to the SQLite file (will be created if absent).
        state: Internal state snapshot from ``Experiment``.
    """
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_DDL)
        with con:
            cur = con.cursor()

            # ── Experiment row ────────────────────────────────────────────────
            cur.execute(
                "INSERT INTO Experiments(name, description) VALUES (?, ?)",
                (state.name, state.description),
            )
            exp_id: int = cur.lastrowid  # type: ignore[assignment]

            # ── Components ───────────────────────────────────────────────────
            comp_version_ids: dict[str, int] = {}  # component.name → version_id
            comp_ids: dict[str, int] = {}  # component.name → component_id
            for comp in state.components:
                cid, vid = _upsert_component(cur, comp.name, comp.type.value, comp.code_hash())
                comp_ids[comp.name] = cid
                comp_version_ids[comp.name] = vid

            # ── ParameterSpecs ───────────────────────────────────────────────
            for p in state.parameters:
                cid = comp_ids.get(p.component.name) if p.component else None
                cur.execute(
                    "INSERT INTO ParameterSpecs(experiment_id, name, values_json, is_static, component_id, conditions_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        exp_id,
                        p.name,
                        _json_stable(p.values),
                        int(p.is_static),
                        cid,
                        _json_stable(p.conditions),
                    ),
                )

            # ── MetricSpecs ──────────────────────────────────────────────────
            for m in state.metrics:
                cur.execute(
                    "INSERT OR IGNORE INTO MetricSpecs(experiment_id, name, type, frequency) VALUES (?, ?, ?, ?)",
                    (exp_id, m.name, m.type.value, m.frequency.value),
                )

            # ── Generate runs ─────────────────────────────────────────────────
            configs = _generate_configs(state)
            _insert_runs(cur, exp_id, comp_version_ids, configs, state)
    finally:
        con.close()


# ── Config generation ─────────────────────────────────────────────────────────

def _generate_configs(state: "_ExperimentState") -> list[dict[str, object]]:
    """Return all valid parameter combinations (cartesian product with conditionals).

    Conditionals are resolved iteratively: after each new param is fixed we
    check whether any conditional parameters become applicable.
    """
    from itertools import product as _product

    # Split parameters into unconditional and conditional groups
    unconditional = [p for p in state.parameters if not p.conditions]
    conditional = [p for p in state.parameters if p.conditions]

    # Start with the cartesian product of unconditional params
    if not unconditional:
        base_configs: list[dict[str, object]] = [{}]
    else:
        keys = [p.name for p in unconditional]
        values_lists = [p.values for p in unconditional]
        base_configs = [dict(zip(keys, combo, strict=True)) for combo in _product(*values_lists)]

    # Expand each base config with any matching conditional params
    expanded: list[dict[str, object]] = []
    for config in base_configs:
        expanded.extend(_expand_conditionals(config, conditional))

    return expanded


def _expand_conditionals(
    config: dict[str, object],
    conditionals: list,
) -> list[dict[str, object]]:
    """Recursively expand conditional parameters triggered by ``config``."""
    from itertools import product as _product

    # Find all conditional params whose conditions are satisfied
    active = [p for p in conditionals if all(config.get(k) == v for k, v in p.conditions.items())]

    if not active:
        return [config]

    # Remaining conditionals (those not yet triggered)
    remaining = [p for p in conditionals if p not in active]

    # Cartesian product over the newly active params
    keys = [p.name for p in active]
    values_lists = [p.values for p in active]
    results: list[dict[str, object]] = []
    for combo in _product(*values_lists):
        merged = {**config, **dict(zip(keys, combo, strict=True))}
        # Recurse: newly fixed values may trigger further conditionals
        results.extend(_expand_conditionals(merged, remaining))

    return results


def _insert_runs(
    cur: sqlite3.Cursor,
    exp_id: int,
    comp_version_ids: dict[str, int],
    configs: list[dict[str, object]],
    state: "_ExperimentState",
) -> None:
    """Insert Run rows for each (config × ablation) combination.

    Seeds are extracted from the ``seed`` key in the config; all other keys
    form the ``HyperparamConfig``.
    """
    # Determine algo/env version ids (first ALGO / ENV component wins)
    algo_vid: int | None = None
    env_vid: int | None = None
    for comp in state.components:
        from .component import ComponentType

        if algo_vid is None and comp.type == ComponentType.ALGO:
            algo_vid = comp_version_ids.get(comp.name)
        if env_vid is None and comp.type == ComponentType.ENV:
            env_vid = comp_version_ids.get(comp.name)

    ablation_list: list[tuple[str | None, dict[str, object]]] = [(None, {})]
    for abl in state.ablations:
        ablation_list.append((abl.name, abl.overrides))

    for config in configs:
        seed = int(config.get("seed", 0))  # type: ignore[arg-type]
        hyper_config = {k: v for k, v in config.items() if k != "seed"}

        for abl_name, abl_overrides in ablation_list:
            final_hyper = {**hyper_config, **abl_overrides}
            hyper_id = _upsert_hyperparam_config(cur, final_hyper)

            cur.execute(
                "INSERT OR IGNORE INTO Runs(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exp_id, algo_vid, env_vid, hyper_id, seed, abl_name),
            )
