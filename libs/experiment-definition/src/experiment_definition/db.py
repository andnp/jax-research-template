"""SQLite persistence layer implementing the ADR 008 schema.

All DDL and upsert logic lives here so ``experiment.py`` stays free of SQL.
Also exposes the standalone ``DatabaseManager`` class for direct
programmatic access without the Fluent API.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import _ExperimentState

from . import _definition_repository as _definition
from ._database_utils import _hash_dict, _json_stable  # noqa: F401
from ._types import PlannedExecution, RunBatch
from .parameter import ParameterValue
from .records import (
    ComponentRow,
    ComponentVersionProvenanceRow,
    ComponentVersionRow,
    ExecutionArtifactRow,
    ExecutionRow,
    ExperimentRow,
    HyperparamConfigRow,
    RunRow,
    _record_from_row,
)
from .schema import ALL_DDL, DEFAULT_DB_NAME

_EXECUTION_TRANSITIONS: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"RUNNING", "COMPLETED", "FAILED", "INVALID"}),
    "RUNNING": frozenset({"COMPLETED", "FAILED", "INVALID"}),
    "COMPLETED": frozenset({"INVALID"}),
    "FAILED": frozenset({"INVALID"}),
    "INVALID": frozenset(),
}


def sync_to_db(
    db_path: Path | str,
    state: "_ExperimentState",
    *,
    static_parameter_names: frozenset[str] | None = None,
):
    """Materialise the experiment definition into a SQLite file.

    Creates the database if it does not exist and applies the ADR 008 schema.
    All operations are wrapped in a single transaction.

    Returns the ids of every run this definition wants, so a caller can identify rows an
    earlier definition left behind.

    Args:
        db_path: Path to the SQLite file (will be created if absent).
        state: Internal state snapshot from ``Experiment``.
    """
    wanted: set[int] = set()
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        with con:
            for ddl in ALL_DDL:
                con.execute(ddl)
            cur = con.cursor()

            # ── Experiment row ────────────────────────────────────────────────
            exp_id = _definition._upsert_experiment(cur, state.name, state.description)

            # ── Components ───────────────────────────────────────────────────
            comp_version_ids: dict[str, int] = {}  # component.name → version_id
            for comp in state.components:
                _, vid = _definition._upsert_component(cur, comp.name, comp.type.value, comp.code_hash())
                comp_version_ids[comp.name] = vid

            # ── Generate runs ─────────────────────────────────────────────────
            configs = _generate_configs(state)
            wanted = _definition.sync_runs(
                cur,
                exp_id,
                comp_version_ids,
                configs,
                state,
                static_parameter_names=static_parameter_names,
            )
    finally:
        con.close()
    return frozenset(wanted)


# ── Config generation ─────────────────────────────────────────────────────────

def _generate_configs(state: "_ExperimentState") -> list[dict[str, ParameterValue]]:
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
        base_configs: list[dict[str, ParameterValue]] = [{}]
    else:
        keys = [p.name for p in unconditional]
        values_lists = [p.values for p in unconditional]
        base_configs = [dict(zip(keys, combo, strict=True)) for combo in _product(*values_lists)]

    # Expand each base config with any matching conditional params
    expanded: list[dict[str, ParameterValue]] = []
    for config in base_configs:
        expanded.extend(_expand_conditionals(config, conditional))

    return expanded


def _expand_conditionals(
    config: dict[str, ParameterValue],
    conditionals: list,
) -> list[dict[str, ParameterValue]]:
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
    results: list[dict[str, ParameterValue]] = []
    for combo in _product(*values_lists):
        merged = {**config, **dict(zip(keys, combo, strict=True))}
        # Recurse: newly fixed values may trigger further conditionals
        results.extend(_expand_conditionals(merged, remaining))

    return results


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
            return _definition.add_component(self.conn, name, component_type)

    def get_component(self, name: str) -> ComponentRow | None:
        """Fetch a component by name."""
        return _definition.get_component(self.conn, name)

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
            return _definition.add_component_version(self.conn, component_id, code_snapshot_hash, notes)

    def get_latest_version(self, component_id: int) -> ComponentVersionRow | None:
        """Resolve the "latest" virtual pointer for a component.

        Returns the ``ComponentVersions`` row with the highest
        ``version_number`` for the given component.  Returns ``None`` if no
        versions exist yet.

        Args:
            component_id: FK into ``Components``.
        """
        return _definition.get_latest_version(self.conn, component_id)

    def get_component_version_provenance(self, version_id: int) -> ComponentVersionProvenanceRow | None:
        """Fetch component identity and provenance for one version."""
        return _definition.get_component_version_provenance(self.conn, version_id)

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
        params: Mapping[str, ParameterValue],
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
        with self.conn:
            return _definition.add_hyperparam_config(self.conn, params, vmap_zone)

    def get_hyperparam_config(self, config_id: int) -> HyperparamConfigRow | None:
        """Fetch a HyperparamConfig by id."""
        return _definition.get_hyperparam_config(self.conn, config_id)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def add_experiment(self, name: str, description: str | None = None) -> int:
        """Insert a new experiment and return its id."""
        with self.conn:
            return _definition.add_experiment(self.conn, name, description)

    def ensure_experiment(self, name: str, description: str | None = None) -> int:
        """Return the existing experiment id for ``name`` or create it.

        This is the additive-registration primitive needed for declarative
        experiment expansion. Repeated calls with the same name are idempotent.
        """
        existing = _definition.get_experiment(self.conn, name)
        if existing is not None:
            return existing.id
        return self.add_experiment(name, description)

    def get_experiment(self, name: str) -> ExperimentRow | None:
        """Fetch an experiment by name."""
        return _definition.get_experiment(self.conn, name)

    def list_experiments(self) -> list[ExperimentRow]:
        """Return all experiments ordered by name."""
        return _definition.list_experiments(self.conn)

    def get_experiment_by_id(self, experiment_id: int) -> ExperimentRow | None:
        """Fetch an experiment by id."""
        return _definition.get_experiment_by_id(self.conn, experiment_id)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def add_run(
        self,
        experiment_id: int,
        algo_version_id: int | None,
        env_version_id: int | None,
        hyper_id: int,
        seed: int,
        ablation: str | None = None,
    ) -> int:
        """Insert a logical Run (intent record) and return its id.

        Raises ``sqlite3.IntegrityError`` if an identical run already exists.
        """
        with self.conn:
            return _definition.add_run(self.conn, experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation)

    def list_runs(self, experiment_id: int) -> list[RunRow]:
        """Return all logical runs for an experiment ordered by seed and id."""
        return _definition.list_runs(self.conn, experiment_id)

    def list_unsatisfied_runs(self, experiment_id: int) -> list[RunRow]:
        """Return runs lacking any linked ``COMPLETED`` execution.

        A run is considered satisfied only when at least one linked execution
        has status ``COMPLETED``. Runs linked only to ``FAILED`` or ``INVALID``
        executions are returned here and should be rescheduled by the runner.
        """
        return _definition.list_unsatisfied_runs(self.conn, experiment_id)

    def list_orphan_runs(self, experiment_id: int, wanted_run_ids: frozenset[int]) -> list[RunRow]:
        """Return runs the current definition does not want and that no execution ever touched.

        ``wanted_run_ids`` comes from ``Experiment.sync``. A run with any linked execution
        is excluded whatever its status: a COMPLETED one is a result, and a PENDING or
        RUNNING one is a live claim, so neither is garbage even when the definition has
        moved on.
        """
        return _definition.list_orphan_runs(self.conn, experiment_id, wanted_run_ids)

    def delete_runs(self, run_ids: list[int]) -> int:
        """Delete runs by id, skipping any that an execution touched, and return the count.

        The execution guard is repeated here rather than trusted from the caller: this
        deletes rows from a durable results database, and a stale id list must not be able
        to remove a run that has results.
        """
        with self.conn:
            return _definition.delete_runs(self.conn, run_ids)

    def list_unsatisfied_run_batches(
        self,
        experiment_id: int,
        max_runs_per_batch: int | None = None,
        only: Mapping[str, str] | None = None,
    ) -> list[RunBatch]:
        """Group unsatisfied runs by static vmap zone for batch planning.

        ``only`` narrows the runs considered, per :func:`_run_matches`. A batch keeps only
        its selected runs rather than being skipped when the selection is partial: a batch
        is a vmap zone, so narrowing one changes how wide it trains but not the identity of
        the runs in it, and skipping would make a single-seed selector silently do nothing.
        """
        rows = self.conn.execute(
            "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation, "
            "       hc.json_blob, hc.vmap_zone_json "
            "FROM Runs r "
            "INNER JOIN HyperparamConfigs hc ON hc.id = r.hyper_id "
            "LEFT JOIN ("
            "    SELECT DISTINCT er.run_id AS run_id "
            "    FROM ExecutionRuns er "
            "    INNER JOIN Executions e ON e.id = er.execution_id "
            "    WHERE e.status IN ('PENDING', 'RUNNING', 'COMPLETED')"
            ") claimed ON claimed.run_id = r.id "
            "WHERE r.experiment_id = ? AND claimed.run_id IS NULL "
            "ORDER BY r.hyper_id, r.seed, r.id",
            (experiment_id,),
        ).fetchall()

        grouped_runs: dict[tuple[int | None, int | None, str | None, str, str | None], list[int]] = {}
        for row in rows:
            run = _record_from_row(RunRow, row)
            hyper_json = str(row[7])
            if only and not _run_matches(json.loads(hyper_json), run, only):
                continue
            vmap_zone_json = str(row[8]) if row[8] is not None else None
            static_config_json = _static_config_json(hyper_json, vmap_zone_json)
            batch_key = (
                run.algo_version_id,
                run.env_version_id,
                run.ablation,
                static_config_json,
                vmap_zone_json,
            )
            grouped_runs.setdefault(batch_key, []).append(run.id)

        def batch_key_of(
            item: tuple[tuple[int | None, int | None, str | None, str, str | None], list[int]],
        ) -> tuple[int | None, int | None, str | None, str, str | None]:
            return item[0]

        batches: list[RunBatch] = []
        for batch_key, run_ids in sorted(grouped_runs.items(), key=batch_key_of):
            algo_version_id, env_version_id, ablation, static_config_json, vmap_zone_json = batch_key
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
                    )
                )
        return batches

    # ------------------------------------------------------------------
    # Executions
    # ------------------------------------------------------------------

    def add_execution(
        self,
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: Mapping[str, ParameterValue] | None = None,
    ) -> int:
        """Create a new Execution record in PENDING status and return its id.

        Args:
            hostname: Machine identifier (e.g. from ``socket.gethostname()``).
            git_commit: Full SHA of the HEAD commit at execution time.
            git_diff_blob: Output of ``git diff HEAD`` for uncommitted changes.
            jax_config: Runtime JAX metadata dict, e.g.::

                    {"jax_version": "0.4.30", "platform": "gpu", "devices": 4}
        """
        with self.conn:
            return self._insert_execution(hostname, git_commit, git_diff_blob, jax_config)

    def _insert_execution(
        self,
        hostname: str | None,
        git_commit: str | None,
        git_diff_blob: str | None,
        jax_config: Mapping[str, ParameterValue] | None,
    ) -> int:
        jax_json = json.dumps(jax_config, sort_keys=True) if jax_config is not None else None
        cur = self.conn.execute(
            "INSERT INTO Executions(hostname, git_commit, git_diff_blob, jax_config_json) "
            "VALUES (?, ?, ?, ?)",
            (hostname, git_commit, git_diff_blob, jax_json),
        )
        if cur.lastrowid is None:
            raise RuntimeError("Insert failed: no execution ID returned.")
        return int(cur.lastrowid)

    def record_execution_git_metadata(
        self,
        execution_id: int,
        git_commit: str | None,
        git_diff_blob: str | None,
    ) -> None:
        """Update git metadata for an execution, keeping existing values on ``None``."""
        with self.conn:
            self.conn.execute(
                "UPDATE Executions SET git_commit = COALESCE(?, git_commit), "
                "git_diff_blob = COALESCE(?, git_diff_blob) WHERE id = ?",
                (git_commit, git_diff_blob, execution_id),
            )

    def get_execution(self, execution_id: int) -> ExecutionRow | None:
        """Fetch a single Execution by id."""
        row = self.conn.execute(
            "SELECT id, status, hostname, start_time, end_time, git_commit, git_diff_blob, jax_config_json "
            "FROM Executions WHERE id = ?",
            (execution_id,),
        ).fetchone()
        return _record_from_row(ExecutionRow, row) if row else None

    def record_execution_artifacts(
        self,
        execution_id: int,
        root_path: str,
        *,
        manifest_path: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Create or update canonical artifact linkage for an execution."""
        with self.conn:
            self._record_execution_artifacts(execution_id, root_path, manifest_path, metadata)

    def _record_execution_artifacts(
        self,
        execution_id: int,
        root_path: str,
        manifest_path: str | None,
        metadata: dict[str, object] | None,
    ) -> None:
        existing = self.conn.execute(
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
            merged_metadata.update(metadata)
        metadata_json = json.dumps(merged_metadata, sort_keys=True) if merged_metadata else None
        self.conn.execute(
            "INSERT INTO ExecutionArtifacts(execution_id, root_path, manifest_path, metadata_json) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(execution_id) DO UPDATE SET "
            "root_path = excluded.root_path, "
            "manifest_path = excluded.manifest_path, "
            "metadata_json = excluded.metadata_json",
            (execution_id, root_path, manifest_path, metadata_json),
        )

    def get_execution_artifacts(self, execution_id: int) -> ExecutionArtifactRow | None:
        """Fetch canonical artifact linkage for an execution."""
        row = self.conn.execute(
            "SELECT execution_id, root_path, manifest_path, metadata_json, created_at "
            "FROM ExecutionArtifacts WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        return _record_from_row(ExecutionArtifactRow, row) if row else None

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
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT status FROM Executions WHERE id = ?", (execution_id,)).fetchone()
            if row is None:
                raise ValueError(f"Execution {execution_id} does not exist.")
            current_status = str(row[0])
            if status not in _EXECUTION_TRANSITIONS[current_status]:
                raise ValueError(f"Illegal execution transition: {current_status} -> {status}.")
            updated = self.conn.execute(
                "UPDATE Executions SET status = ?, start_time = COALESCE(?, start_time), "
                "end_time = COALESCE(?, end_time) WHERE id = ? AND status = ?",
                (status, start_time, end_time, execution_id, current_status),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Execution {execution_id} changed before it could be updated.")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def reclaim_stale_executions(
        self,
        experiment_id: int,
        stale_after_seconds: float,
    ) -> list[int]:
        """Fail RUNNING executions of ``experiment_id`` orphaned by a dead process.

        A run is claimed by any linked PENDING/RUNNING/COMPLETED execution
        (see ``list_unsatisfied_run_batches``), and is only satisfied by a
        COMPLETED one. A process that dies without unwinding — SIGKILL, OOM,
        or its working directory disappearing out from under it — never
        reaches the ``except`` handler in ``execute_batch`` that would mark
        the execution FAILED, so it stays RUNNING forever and permanently
        blocks its runs from being replanned or satisfied.

        Liveness signal: ``start_time`` staleness, not a recorded PID. A PID
        can only be probed on the same host as the recorder (useless for a
        resume from a different machine, and reused PIDs can lie), and
        recording one would require a new column on the ``Executions`` table.
        ``start_time`` already exists, is already written by
        ``execute_batch``, and needs no schema change. The tradeoff is that
        a threshold cannot distinguish "dead" from "slow but alive", so this
        method is opt-in: callers must pass a ``stale_after_seconds`` known
        to safely exceed the longest real batch's runtime, and a currently
        RUNNING execution younger than the threshold is never touched.

        Returns:
            The ids of executions transitioned to FAILED.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
        with self.conn:
            rows = self.conn.execute(
                "SELECT DISTINCT e.id FROM Executions e "
                "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
                "INNER JOIN Runs r ON r.id = er.run_id "
                "WHERE r.experiment_id = ? AND e.status = 'RUNNING' "
                "AND e.start_time IS NOT NULL AND e.start_time < ?",
                (experiment_id, cutoff),
            ).fetchall()
        reclaimed: list[int] = []
        for row in rows:
            execution_id = int(row[0])
            self.update_execution_status(
                execution_id, "FAILED", end_time=datetime.now(timezone.utc).isoformat()
            )
            reclaimed.append(execution_id)
        return reclaimed

    def plan_execution(
        self,
        run_ids: list[int],
        *,
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: Mapping[str, ParameterValue] | None = None,
    ) -> int:
        """Create a pending execution and link it to the supplied runs."""
        if not run_ids:
            raise ValueError("run_ids must not be empty.")

        with self.conn:
            return self._plan_execution(run_ids, hostname, git_commit, git_diff_blob, jax_config)

    def _plan_execution(
        self,
        run_ids: list[int],
        hostname: str | None,
        git_commit: str | None,
        git_diff_blob: str | None,
        jax_config: Mapping[str, ParameterValue] | None,
    ) -> int:
        execution_id = self._insert_execution(hostname, git_commit, git_diff_blob, jax_config)
        for run_id in run_ids:
            self._link_execution_run(execution_id, run_id)
        return execution_id

    def plan_unsatisfied_execution(
        self,
        experiment_id: int,
        *,
        limit: int | None = None,
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: Mapping[str, ParameterValue] | None = None,
    ) -> int | None:
        """Create a pending execution covering unsatisfied runs for an experiment."""
        rows = self.conn.execute(
            "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation "
            "FROM Runs r "
            "LEFT JOIN ("
            "    SELECT DISTINCT er.run_id AS run_id "
            "    FROM ExecutionRuns er "
            "    INNER JOIN Executions e ON e.id = er.execution_id "
            "    WHERE e.status IN ('PENDING', 'RUNNING', 'COMPLETED')"
            ") claimed ON claimed.run_id = r.id "
            "WHERE r.experiment_id = ? AND claimed.run_id IS NULL "
            "ORDER BY r.seed, r.hyper_id, r.id",
            (experiment_id,),
        ).fetchall()
        if not rows:
            return None

        unsatisfied_runs = [_record_from_row(RunRow, row) for row in rows]
        selected_runs = unsatisfied_runs if limit is None else unsatisfied_runs[:limit]
        return self.plan_execution(
            [run.id for run in selected_runs],
            hostname=hostname,
            git_commit=git_commit,
            git_diff_blob=git_diff_blob,
            jax_config=jax_config,
        )

    def plan_unsatisfied_execution_batches(
        self,
        experiment_id: int,
        *,
        max_runs_per_batch: int | None = None,
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: Mapping[str, ParameterValue] | None = None,
    ) -> list[int]:
        """Create one pending execution per unsatisfied static-config batch."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            execution_ids: list[int] = []
            for batch in self.list_unsatisfied_run_batches(experiment_id, max_runs_per_batch=max_runs_per_batch):
                execution_ids.append(
                    self._plan_execution(batch.run_ids, hostname, git_commit, git_diff_blob, jax_config)
                )
            self.conn.commit()
            return execution_ids
        except Exception:
            self.conn.rollback()
            raise

    def plan_experiment_execution_batches(
        self,
        experiment_id: int,
        artifacts_root: str | Path,
        *,
        max_runs_per_batch: int | None = None,
        manifest_name: str = "manifest.json",
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: Mapping[str, ParameterValue] | None = None,
    ) -> list[PlannedExecution]:
        """Plan execution batches and register deterministic artifact paths."""
        base_path = Path(artifacts_root)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            planned: list[PlannedExecution] = []
            for batch in self.list_unsatisfied_run_batches(experiment_id, max_runs_per_batch=max_runs_per_batch):
                execution_id = self._plan_execution(batch.run_ids, hostname, git_commit, git_diff_blob, jax_config)
                root_path = str(base_path / str(execution_id))
                manifest_path = str(Path(root_path) / manifest_name)
                self._record_execution_artifacts(
                    execution_id,
                    root_path,
                    manifest_path,
                    {
                        "run_ids": batch.run_ids,
                        "static_config_json": _json_stable(batch.static_config),
                        "vmap_zone_json": _json_stable(batch.vmap_zone) if batch.vmap_zone is not None else None,
                    },
                )
                planned.append(
                    PlannedExecution(execution_id, batch.run_ids, root_path, manifest_path, batch.static_config, batch.vmap_zone)
                )
            self.conn.commit()
            return planned
        except Exception:
            self.conn.rollback()
            raise

    def plan_next_execution_batch(
        self,
        experiment_id: int,
        artifacts_root: str | Path,
        *,
        max_runs_per_batch: int | None = None,
        manifest_name: str = "manifest.json",
        hostname: str | None = None,
        git_commit: str | None = None,
        git_diff_blob: str | None = None,
        jax_config: Mapping[str, ParameterValue] | None = None,
        only: Mapping[str, str] | None = None,
    ) -> PlannedExecution | None:
        """Plan and return the next unsatisfied execution batch, or None when none remain.

        ``only`` restricts which runs are eligible, per :func:`_run_matches`.
        """
        base_path = Path(artifacts_root)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            batches = self.list_unsatisfied_run_batches(experiment_id, max_runs_per_batch=max_runs_per_batch, only=only)
            if not batches:
                self.conn.commit()
                return None
            batch = batches[0]
            execution_id = self._plan_execution(batch.run_ids, hostname, git_commit, git_diff_blob, jax_config)
            root_path = str(base_path / str(execution_id))
            manifest_path = str(Path(root_path) / manifest_name)
            self._record_execution_artifacts(
                execution_id,
                root_path,
                manifest_path,
                {
                    "run_ids": batch.run_ids,
                    "static_config_json": _json_stable(batch.static_config),
                    "vmap_zone_json": _json_stable(batch.vmap_zone) if batch.vmap_zone is not None else None,
                },
            )
            planned = PlannedExecution(execution_id, batch.run_ids, root_path, manifest_path, batch.static_config, batch.vmap_zone)
            self.conn.commit()
            return planned
        except Exception:
            self.conn.rollback()
            raise

    def list_execution_runs(self, execution_id: int) -> list[RunRow]:
        """Return the logical runs linked to an execution."""
        rows = self.conn.execute(
            "SELECT r.id, r.experiment_id, r.algo_version_id, r.env_version_id, r.hyper_id, r.seed, r.ablation "
            "FROM Runs r "
            "INNER JOIN ExecutionRuns er ON er.run_id = r.id "
            "WHERE er.execution_id = ? "
            "ORDER BY r.seed, r.hyper_id, r.id",
            (execution_id,),
        ).fetchall()
        return [_record_from_row(RunRow, row) for row in rows]

    def get_latest_completed_execution_for_run(self, run_id: int) -> ExecutionRow | None:
        """Return the latest completed execution linked to a run, if any."""
        row = self.conn.execute(
            "SELECT e.id, e.status, e.hostname, e.start_time, e.end_time, e.git_commit, e.git_diff_blob, e.jax_config_json "
            "FROM Executions e "
            "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
            "WHERE er.run_id = ? AND e.status = 'COMPLETED' "
            "ORDER BY COALESCE(e.end_time, e.start_time, '') DESC, e.id DESC "
            "LIMIT 1",
            (run_id,),
        ).fetchone()
        return _record_from_row(ExecutionRow, row) if row else None

    def get_latest_completed_artifacts_for_run(self, run_id: int) -> ExecutionArtifactRow | None:
        """Resolve artifact linkage for the latest completed execution of a run."""
        latest_execution = self.get_latest_completed_execution_for_run(run_id)
        if latest_execution is None:
            return None
        return self.get_execution_artifacts(latest_execution.id)

    def link_execution_run(self, execution_id: int, run_id: int) -> None:
        """Record that an Execution covers a logical Run (ExecutionRuns bridge)."""
        with self.conn:
            self._link_execution_run(execution_id, run_id)

    def _link_execution_run(self, execution_id: int, run_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO ExecutionRuns(execution_id, run_id) VALUES (?, ?)",
            (execution_id, run_id),
        )

    def invalidate_execution(self, execution_id: int):
        existing = self.get_execution(execution_id)
        if existing is None:
            return False
        self.update_execution_status(execution_id, "INVALID")
        return True

    def invalidate_executions_by_commit(self, git_commit: str):
        with self.conn:
            rows = self.conn.execute(
                "SELECT id FROM Executions WHERE git_commit = ? AND status IN ('COMPLETED', 'FAILED')",
                (git_commit,),
            ).fetchall()
            if not rows:
                return []
            self.conn.execute(
                "UPDATE Executions SET status = 'INVALID' WHERE git_commit = ? AND status IN ('COMPLETED', 'FAILED')",
                (git_commit,),
            )
        return [int(row[0]) for row in rows]

    def find_executions_for_experiment(self, experiment_id: int) -> list[ExecutionRow]:
        """Return every COMPLETED/FAILED execution covering a run of ``experiment_id``.

        The whole-experiment selector, for a deliberate clean re-run rather than a
        targeted correction. Unlike ``find_executions_by_hyperparams`` there is no
        per-batch filter to under-select against: the caller asked for all of them.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT e.id, e.status, e.hostname, e.start_time, e.end_time, "
            "e.git_commit, e.git_diff_blob, e.jax_config_json "
            "FROM Executions e "
            "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
            "INNER JOIN Runs r ON r.id = er.run_id "
            "WHERE r.experiment_id = ? AND e.status IN ('COMPLETED', 'FAILED') "
            "ORDER BY e.id",
            (experiment_id,),
        ).fetchall()
        return [_record_from_row(ExecutionRow, row) for row in rows]

    def find_executions_by_hyperparams(self, experiment_id: int, where: Mapping[str, str]) -> list[ExecutionRow]:
        """Return COMPLETED/FAILED executions whose covered runs all match ``where``.

        Each clause compares the CLI string against ``str()`` of the
        JSON-decoded hyperparameter value, so ``learning_rate=0.001`` matches
        a stored float ``0.001`` but not an equal value spelled ``1e-3``. An
        execution batch is included only when every run it covers satisfies
        every clause -- a partially matching batch is skipped, since
        invalidating runs the caller did not select is worse than
        under-selecting.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT e.id, e.status, e.hostname, e.start_time, e.end_time, "
            "e.git_commit, e.git_diff_blob, e.jax_config_json "
            "FROM Executions e "
            "INNER JOIN ExecutionRuns er ON er.execution_id = e.id "
            "INNER JOIN Runs r ON r.id = er.run_id "
            "WHERE r.experiment_id = ? AND e.status IN ('COMPLETED', 'FAILED') "
            "ORDER BY e.id",
            (experiment_id,),
        ).fetchall()
        matched = []
        for row in rows:
            execution = _record_from_row(ExecutionRow, row)
            runs = self.list_execution_runs(execution.id)
            if runs and all(self._hyperparams_match(run.hyper_id, where) for run in runs):
                matched.append(execution)
        return matched

    def _hyperparams_match(self, hyper_id: int, where: Mapping[str, str]) -> bool:
        config = self.get_hyperparam_config(hyper_id)
        if config is None:
            return False
        params = config.hyperparameters
        return all(key in params and str(params[key]) == value for key, value in where.items())

    def list_executions(
        self,
        experiment_id: int | None = None,
        *,
        status: str | None = None,
        git_commit: str | None = None,
    ):
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
        rows = self.conn.execute(
            "SELECT e.id, e.status, e.hostname, e.start_time, e.end_time, "
            "e.git_commit, e.git_diff_blob, e.jax_config_json "
            f"FROM Executions e{where} ORDER BY e.id DESC",
            params,
        ).fetchall()
        return [_record_from_row(ExecutionRow, row) for row in rows]

    def get_planned_execution(self, execution_id: int):
        execution = self.get_execution(execution_id)
        if execution is None:
            return None
        artifacts = self.get_execution_artifacts(execution_id)
        if artifacts is None:
            return None
        runs = self.list_execution_runs(execution_id)
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
    """Whether one run satisfies every clause in ``only``.

    Clauses AND together; a clause's comma-separated values OR within it, so
    ``seed=0,1,2`` selects a range that repeated strict-equality clauses could not
    express. Values compare as strings against ``str()`` of the stored value, as
    ``find_executions_by_hyperparams`` does, so ``learning_rate=1e-3`` does not match a
    stored ``0.001``.

    ``seed`` and ``ablation`` live on ``Runs`` rather than in the hyperparameter config,
    and shadow a config key of the same name -- the column is the identity the run was
    planned under.
    """
    fields: dict[str, object] = {**hyperparams, "seed": run.seed, "ablation": run.ablation}
    return all(
        key in fields and str(fields[key]) in set(value.split(","))
        for key, value in only.items()
    )


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
