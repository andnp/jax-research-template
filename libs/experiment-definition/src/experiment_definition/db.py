"""SQLite persistence layer implementing the ADR 008 schema.

All DDL and upsert logic lives here so ``experiment.py`` stays free of SQL.
Also exposes the standalone ``DatabaseManager`` class for direct
programmatic access without the Fluent API.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .experiment import _ExperimentState

from .parameter import ParameterValue
from .schema import ALL_DDL, DEFAULT_DB_NAME


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
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        with con:
            for ddl in ALL_DDL:
                con.execute(ddl)
            cur = con.cursor()

            # ── Experiment row ────────────────────────────────────────────────
            cur.execute(
                "INSERT INTO Experiments(name, description) VALUES (?, ?)",
                (state.name, state.description),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Insert failed: no experiment ID returned.")
            exp_id: int = cur.lastrowid

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

    ablation_list: list[tuple[str | None, dict[str, ParameterValue]]] = [(None, {})]
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


# ---------------------------------------------------------------------------
# Small data carriers (NamedTuple for zero-overhead, fully typed results)
# ---------------------------------------------------------------------------


class ComponentRow(NamedTuple):
    id: int
    name: str
    type: str


class ComponentVersionRow(NamedTuple):
    id: int
    component_id: int
    version_number: int
    created_at: str
    code_snapshot_hash: str
    notes: str | None


class HyperparamConfigRow(NamedTuple):
    id: int
    hash: str
    json_blob: str
    vmap_zone_json: str | None


class ExperimentRow(NamedTuple):
    id: int
    name: str
    description: str | None
    created_at: str


class RunRow(NamedTuple):
    id: int
    experiment_id: int
    algo_version_id: int
    env_version_id: int
    hyper_id: int
    seed: int


class ExecutionRow(NamedTuple):
    id: int
    status: str
    hostname: str | None
    start_time: str | None
    end_time: str | None
    git_commit: str | None
    git_diff_blob: str | None
    jax_config_json: str | None


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Manages the lifecycle of the master experiment SQLite database.

    Args:
        db_path: Path to the SQLite file. Pass ``":memory:"`` for an ephemeral
            in-process database (useful in tests).

    Example:
        >>> db = DatabaseManager("experiments.sqlite")
        >>> db.initialize()
        >>> comp_id = db.add_component("PPO", "ALGO")
        >>> ver_id = db.add_component_version(comp_id, "abc123")
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_NAME) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open (or re-open) the connection and enable WAL + foreign keys."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def close(self) -> None:
        """Flush and close the connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DatabaseManager":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() or use as context manager.")
        return self._conn

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables, indexes, and views if they do not exist.

        Idempotent: safe to call on an already-initialised database.
        """
        with self.conn:
            for ddl in ALL_DDL:
                self.conn.execute(ddl)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def add_component(self, name: str, component_type: str) -> int:
        """Insert a new component and return its id.

        Args:
            name: Unique human-readable name (e.g. ``"PPO"``).
            component_type: One of ``"ALGO"``, ``"ENV"``, ``"WRAPPER"``.

        Returns:
            The integer ``id`` of the newly inserted row.

        Raises:
            sqlite3.IntegrityError: If a component with this name already exists.
            ValueError: If ``component_type`` is not a valid value.
        """
        if component_type not in ("ALGO", "ENV", "WRAPPER"):
            raise ValueError(f"Invalid component_type {component_type!r}. Must be ALGO, ENV, or WRAPPER.")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO Components(name, type) VALUES (?, ?)",
                (name, component_type),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Insert failed: no component ID returned.")
            return int(cur.lastrowid)

    def get_component(self, name: str) -> ComponentRow | None:
        """Fetch a component by name."""
        row = self.conn.execute("SELECT id, name, type FROM Components WHERE name = ?", (name,)).fetchone()
        return ComponentRow(*row) if row else None

    # ------------------------------------------------------------------
    # ComponentVersions  (researcher-in-the-loop auto-increment)
    # ------------------------------------------------------------------

    def add_component_version(
        self,
        component_id: int,
        code_snapshot_hash: str,
        notes: str | None = None,
    ) -> int:
        """Insert a new ComponentVersion with auto-incremented version_number.

        The version_number is scoped *per component* and computed as
        ``MAX(version_number) + 1`` inside a single transaction, preventing
        races between concurrent writers.

        Args:
            component_id: FK into ``Components``.
            code_snapshot_hash: SHA-256 hex digest of the component source.
            notes: Optional researcher note (e.g. ``"Fixed reward normalisation bug"``).

        Returns:
            The integer ``id`` of the newly inserted ``ComponentVersions`` row.
        """
        with self.conn:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) FROM ComponentVersions WHERE component_id = ?",
                (component_id,),
            ).fetchone()
            next_version = int(row[0]) + 1
            cur = self.conn.execute(
                "INSERT INTO ComponentVersions(component_id, version_number, code_snapshot_hash, notes) "
                "VALUES (?, ?, ?, ?)",
                (component_id, next_version, code_snapshot_hash, notes),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Insert failed: no component version ID returned.")
            return int(cur.lastrowid)

    def get_latest_version(self, component_id: int) -> ComponentVersionRow | None:
        """Resolve the "latest" virtual pointer for a component.

        Returns the ``ComponentVersions`` row with the highest
        ``version_number`` for the given component.  Returns ``None`` if no
        versions exist yet.

        Args:
            component_id: FK into ``Components``.
        """
        row = self.conn.execute(
            "SELECT id, component_id, version_number, created_at, code_snapshot_hash, notes "
            "FROM ComponentLatestVersions WHERE component_id = ?",
            (component_id,),
        ).fetchone()
        return ComponentVersionRow(*row) if row else None

    def hash_changed(self, component_id: int, new_hash: str) -> bool:
        """Return ``True`` if ``new_hash`` differs from the latest stored hash.

        Used by the CLI to decide whether to prompt the researcher for a new
        version (ADR 008 §2 Automatic Versioning & Researcher Intent).

        Args:
            component_id: FK into ``Components``.
            new_hash: SHA-256 hex digest of the current source on disk.
        """
        latest = self.get_latest_version(component_id)
        if latest is None:
            return True
        return latest.code_snapshot_hash != new_hash

    # ------------------------------------------------------------------
    # HyperparamConfigs
    # ------------------------------------------------------------------

    def add_hyperparam_config(
        self,
        params: dict[str, object],
        vmap_zone: dict[str, list[str]] | None = None,
    ) -> int:
        """Upsert a hyperparam configuration and return its id.

        The ``hash`` is a SHA-256 digest of the canonical JSON representation,
        making configs content-addressed and deduplicated.

        Args:
            params: Flat dictionary of hyperparameter name → value.
            vmap_zone: Optional JAX vmap metadata::

                    {"static_keys": ["arch"], "dynamic_keys": ["lr", "seed"]}

                ``static_keys`` require recompilation when changed; 
                ``dynamic_keys`` can be batched via ``jax.vmap``.

        Returns:
            The integer ``id`` of the (possibly pre-existing) row.
        """
        json_blob = json.dumps(params, sort_keys=True)
        config_hash = hashlib.sha256(json_blob.encode()).hexdigest()
        vmap_json = json.dumps(vmap_zone, sort_keys=True) if vmap_zone is not None else None

        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO HyperparamConfigs(hash, json_blob, vmap_zone_json) VALUES (?, ?, ?)",
                (config_hash, json_blob, vmap_json),
            )
            row = self.conn.execute("SELECT id FROM HyperparamConfigs WHERE hash = ?", (config_hash,)).fetchone()
            return int(row[0])

    def get_hyperparam_config(self, config_id: int) -> HyperparamConfigRow | None:
        """Fetch a HyperparamConfig by id."""
        row = self.conn.execute(
            "SELECT id, hash, json_blob, vmap_zone_json FROM HyperparamConfigs WHERE id = ?",
            (config_id,),
        ).fetchone()
        return HyperparamConfigRow(*row) if row else None

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def add_experiment(self, name: str, description: str | None = None) -> int:
        """Insert a new experiment and return its id."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO Experiments(name, description) VALUES (?, ?)",
                (name, description),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Insert failed: no experiment ID returned.")
            return int(cur.lastrowid)

    def get_experiment(self, name: str) -> ExperimentRow | None:
        """Fetch an experiment by name."""
        row = self.conn.execute(
            "SELECT id, name, description, created_at FROM Experiments WHERE name = ?",
            (name,),
        ).fetchone()
        return ExperimentRow(*row) if row else None

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def add_run(
        self,
        experiment_id: int,
        algo_version_id: int,
        env_version_id: int,
        hyper_id: int,
        seed: int,
        ablation: str | None = None,
    ) -> int:
        """Insert a logical Run (intent record) and return its id.

        Raises ``sqlite3.IntegrityError`` if an identical run already exists.
        """
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO Runs(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Insert failed: no run ID returned.")
            return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # Executions
    # ------------------------------------------------------------------

    def add_execution(
        self,
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: dict[str, object] | None = None,
    ) -> int:
        """Create a new Execution record in PENDING status and return its id.

        Args:
            hostname: Machine identifier (e.g. from ``socket.gethostname()``).
            git_commit: Full SHA of the HEAD commit at execution time.
            git_diff_blob: Output of ``git diff HEAD`` for uncommitted changes.
            jax_config: Runtime JAX metadata dict, e.g.::

                    {"jax_version": "0.4.30", "platform": "gpu", "devices": 4}
        """
        jax_json = json.dumps(jax_config, sort_keys=True) if jax_config is not None else None
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO Executions(hostname, git_commit, git_diff_blob, jax_config_json) "
                "VALUES (?, ?, ?, ?)",
                (hostname, git_commit, git_diff_blob, jax_json),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Insert failed: no execution ID returned.")
            return int(cur.lastrowid)

    def update_execution_status(
        self,
        execution_id: int,
        status: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """Update the status (and optional timestamps) of an Execution.

        Args:
            execution_id: PK of the Execution to update.
            status: One of ``PENDING``, ``RUNNING``, ``COMPLETED``, ``FAILED``, ``INVALID``.
            start_time: ISO-8601 timestamp string.
            end_time: ISO-8601 timestamp string.
        """
        if status not in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INVALID"):
            raise ValueError(f"Invalid status {status!r}.")
        with self.conn:
            self.conn.execute(
                "UPDATE Executions SET status = ?, start_time = COALESCE(?, start_time), "
                "end_time = COALESCE(?, end_time) WHERE id = ?",
                (status, start_time, end_time, execution_id),
            )

    def link_execution_run(self, execution_id: int, run_id: int) -> None:
        """Record that an Execution covers a logical Run (ExecutionRuns bridge)."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO ExecutionRuns(execution_id, run_id) VALUES (?, ?)",
                (execution_id, run_id),
            )
