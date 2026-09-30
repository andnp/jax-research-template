"""SQLite persistence layer implementing the ADR 008 schema.

All DDL and upsert logic lives here so ``experiment.py`` stays free of SQL.
Also exposes the standalone ``DatabaseManager`` class for direct
programmatic access without the Fluent API.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import _ExperimentState

from . import _definition_repository as _definition
from . import _execution_repository as _execution
from . import _planning_service as _planning
from . import _query_repository as _query
from . import schema
from ._database_utils import _hash_dict  # noqa: F401
from ._types import PlannedExecution, RunBatch
from .parameter import ParameterValue
from .records import (
    ComponentRow,
    ComponentVersionProvenanceRow,
    ExecutionArtifactRow,
    ExecutionRow,
    ExperimentRow,
    HyperparamConfigRow,
    RunRow,
)
from .schema import ALL_DDL, DEFAULT_DB_NAME


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
        con.execute("BEGIN IMMEDIATE")
        with con:
            for ddl in ALL_DDL:
                con.execute(ddl)
            _migrate_run_vmap_zone(con)
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


def _migrate_run_vmap_zone(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(Runs)")}
    if "vmap_zone_json" in columns:
        return
    conn.execute("ALTER TABLE Runs ADD COLUMN vmap_zone_json TEXT")
    conn.execute(
        "UPDATE Runs SET vmap_zone_json = "
        "(SELECT hc.vmap_zone_json FROM HyperparamConfigs hc WHERE hc.id = Runs.hyper_id)"
    )


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
        self._conn.execute("PRAGMA busy_timeout=30000;")

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
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for ddl in ALL_DDL:
                self.conn.execute(ddl)
            _migrate_run_vmap_zone(self.conn)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

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
        if component_type not in schema.COMPONENT_TYPES:
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

    def get_component_version_provenance(self, version_id: int) -> ComponentVersionProvenanceRow | None:
        """Fetch component identity and provenance for one version."""
        return _definition.get_component_version_provenance(self.conn, version_id)

    # ------------------------------------------------------------------
    # HyperparamConfigs
    # ------------------------------------------------------------------

    def add_hyperparam_config(self, params: Mapping[str, ParameterValue]) -> int:
        """Upsert a hyperparam configuration and return its id.

        The ``hash`` is a SHA-256 digest of the canonical JSON representation,
        making configs content-addressed and deduplicated.
        """
        with self.conn:
            return _definition.add_hyperparam_config(self.conn, params)

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
        vmap_zone: dict[str, list[str]] | None = None,
    ) -> int:
        """Insert a logical Run (intent record) and return its id.

        Raises ``sqlite3.IntegrityError`` if an identical run already exists.
        """
        with self.conn:
            return _definition.add_run(
                self.conn, experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation, vmap_zone
            )

    def list_runs(self, experiment_id: int) -> list[RunRow]:
        """Return all logical runs for an experiment ordered by seed and id."""
        return _definition.list_runs(self.conn, experiment_id)

    def list_runs_without_completed_execution(self, experiment_id: int) -> list[RunRow]:
        """Return runs lacking a linked completed execution.

        A run is considered satisfied only when at least one linked execution
        has status ``COMPLETED``. Runs linked only to ``FAILED`` or ``INVALID``
        executions are returned here and should be rescheduled by the runner.
        """
        return _definition.list_runs_without_completed_execution(self.conn, experiment_id)

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
        group: str | None = None,
    ) -> list[RunBatch]:
        """Group unsatisfied runs by static vmap zone for batch planning.

        ``only`` narrows the runs considered, per the planning service matcher. A batch keeps only
        its selected runs rather than being skipped when the selection is partial: a batch
        is a vmap zone, so narrowing one changes how wide it trains but not the identity of
        the runs in it, and skipping would make a single-seed selector silently do nothing.
        """
        return _planning.list_unsatisfied_run_batches(
            self.conn, experiment_id, max_runs_per_batch, only, group
        )

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
            return _execution.add_execution(self.conn, hostname, git_commit, git_diff_blob, jax_config)

    def record_execution_git_metadata(
        self,
        execution_id: int,
        git_commit: str | None,
        git_diff_blob: str | None,
    ) -> None:
        """Update git metadata for an execution, keeping existing values on ``None``."""
        with self.conn:
            _execution.record_execution_git_metadata(self.conn, execution_id, git_commit, git_diff_blob)

    def get_execution(self, execution_id: int) -> ExecutionRow | None:
        """Fetch a single Execution by id."""
        return _execution.get_execution(self.conn, execution_id)

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
            _execution.record_execution_artifacts(self.conn, execution_id, root_path, manifest_path, metadata)

    def get_execution_artifacts(self, execution_id: int) -> ExecutionArtifactRow | None:
        """Fetch canonical artifact linkage for an execution."""
        return _execution.get_execution_artifacts(self.conn, execution_id)

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
        _execution.update_execution_status(self.conn, execution_id, status, start_time, end_time)

    def fail_execution(self, execution_id: int) -> None:
        """Mark a PENDING or RUNNING execution FAILED now, leaving a finished one untouched."""
        _execution.fail_live_execution(self.conn, execution_id, datetime.now(timezone.utc).isoformat())

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
        return _execution.reclaim_stale_executions(self.conn, experiment_id, stale_after_seconds)

    def fail_group_executions_started_since(self, experiment_id: int, group: str, since: datetime) -> list[int]:
        """Fail the RUNNING executions of one group that started at or after ``since``.

        For a worker that died without unwinding: one worker runs a group at a time, so these
        are that worker's executions, and failing them lets the next launch replan their runs
        without a stale-reclaim threshold.
        """
        return _planning.fail_group_executions_started_since(
            self.conn, experiment_id, group, since.astimezone(timezone.utc).isoformat()
        )

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
            for batch in _planning.list_unsatisfied_run_batches(
                self.conn, experiment_id, max_runs_per_batch, None, None
            ):
                planned.append(
                    _planning.plan_batch_with_artifacts(
                        self.conn,
                        batch,
                        base_path,
                        manifest_name,
                        hostname,
                        git_commit,
                        git_diff_blob,
                        jax_config,
                    )
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
        group: str | None = None,
    ) -> PlannedExecution | None:
        """Plan and return the next unsatisfied execution batch, or None when none remain.

        ``only`` restricts which runs are eligible, per the planning service matcher; ``group``
        restricts planning to the batches whose ``RunBatch.group_key`` matches.
        """
        base_path = Path(artifacts_root)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            batches = _planning.list_unsatisfied_run_batches(
                self.conn, experiment_id, max_runs_per_batch, only, group
            )
            if not batches:
                self.conn.commit()
                return None
            batch = batches[0]
            planned = _planning.plan_batch_with_artifacts(
                self.conn,
                batch,
                base_path,
                manifest_name,
                hostname,
                git_commit,
                git_diff_blob,
                jax_config,
            )
            self.conn.commit()
            return planned
        except Exception:
            self.conn.rollback()
            raise

    def list_execution_runs(self, execution_id: int) -> list[RunRow]:
        """Return the logical runs linked to an execution."""
        return _query.list_execution_runs(self.conn, execution_id)

    def get_latest_completed_execution_for_run(self, run_id: int) -> ExecutionRow | None:
        """Return the latest completed execution linked to a run, if any."""
        return _query.get_latest_completed_execution_for_run(self.conn, run_id)

    def list_run_execution_statuses(self, experiment_id: int) -> dict[int, frozenset[str]]:
        """Return every execution status linked to each of an experiment's runs, empty when never executed."""
        return _query.list_run_execution_statuses(self.conn, experiment_id)

    def link_execution_run(self, execution_id: int, run_id: int) -> None:
        """Record that an Execution covers a logical Run (ExecutionRuns bridge)."""
        with self.conn:
            self._link_execution_run(execution_id, run_id)

    def _link_execution_run(self, execution_id: int, run_id: int) -> None:
        _execution._link_execution_run(self.conn, execution_id, run_id)

    def invalidate_execution(self, execution_id: int):
        return _query.invalidate_execution(self.conn, execution_id)

    def invalidate_executions_by_commit(self, git_commit: str):
        return _query.invalidate_executions_by_commit(self.conn, git_commit)

    def find_executions_for_experiment(self, experiment_id: int) -> list[ExecutionRow]:
        """Return every COMPLETED/FAILED execution covering a run of ``experiment_id``.

        The whole-experiment selector, for a deliberate clean re-run rather than a
        targeted correction. Unlike ``find_executions_by_hyperparams`` there is no
        per-batch filter to under-select against: the caller asked for all of them.
        """
        return _query.find_executions_for_experiment(self.conn, experiment_id)

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
        return _query.find_executions_by_hyperparams(self.conn, experiment_id, where)


    def list_executions(
        self,
        experiment_id: int | None = None,
        *,
        status: str | None = None,
        git_commit: str | None = None,
    ):
        return _query.list_executions(
            self.conn,
            experiment_id,
            status=status,
            git_commit=git_commit,
        )

    def get_planned_execution(self, execution_id: int):
        return _query.get_planned_execution(self.conn, execution_id)
