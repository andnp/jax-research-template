"""Medium (< 1 s) integration tests for the experiment-definition DatabaseManager.

These tests exercise the actual SQLite interactions using an in-memory database,
verifying schema creation, CRUD operations, auto-versioning, and the "latest"
version pointer logic.
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest
from experiment_definition.db import DatabaseManager


def _require[T](value: T | None) -> T:
    """Return a looked-up row, failing the test if it was not found.

    The db lookups return ``| None`` because a miss is legitimate. These tests
    seed the rows first, so a miss means the fixture is broken -- assert that
    directly instead of tripping over ``None`` further downstream.
    """
    assert value is not None
    return value

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> DatabaseManager:
    """Return an initialised in-memory DatabaseManager."""
    manager = DatabaseManager(":memory:")
    manager.connect()
    manager.initialize()
    return manager


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def test_initialize_creates_all_tables(db: DatabaseManager) -> None:
    tables = {
        row[0]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    required = {"Components", "ComponentVersions", "HyperparamConfigs", "Experiments", "Runs", "Executions", "ExecutionRuns"}
    assert required.issubset(tables)


def test_initialize_creates_latest_versions_view(db: DatabaseManager) -> None:
    views = {
        row[0]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
    }
    assert "ComponentLatestVersions" in views


def test_initialize_is_idempotent(db: DatabaseManager) -> None:
    """Calling initialize() twice must not raise."""
    db.initialize()


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_add_component_returns_id(db: DatabaseManager) -> None:
    cid = db.add_component("PPO", "ALGO")
    assert isinstance(cid, int) and cid > 0


def test_add_component_invalid_type_raises(db: DatabaseManager) -> None:
    with pytest.raises(ValueError, match="Invalid component_type"):
        db.add_component("BadComp", "MODEL")


def test_add_component_duplicate_raises(db: DatabaseManager) -> None:
    db.add_component("DQN", "ALGO")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_component("DQN", "ALGO")


def test_get_component_returns_row(db: DatabaseManager) -> None:
    db.add_component("CartPole", "ENV")
    row = db.get_component("CartPole")
    assert row is not None
    assert row.name == "CartPole"
    assert row.type == "ENV"


def test_get_component_missing_returns_none(db: DatabaseManager) -> None:
    assert db.get_component("NonExistent") is None


# ---------------------------------------------------------------------------
# ComponentVersions — auto-increment and researcher-in-the-loop
# ---------------------------------------------------------------------------


def test_first_version_is_one(db: DatabaseManager) -> None:
    cid = db.add_component("SAC", "ALGO")
    vid = db.add_component_version(cid, "hash_v1")
    row = db.conn.execute("SELECT version_number FROM ComponentVersions WHERE id = ?", (vid,)).fetchone()
    assert row[0] == 1


def test_second_version_increments(db: DatabaseManager) -> None:
    cid = db.add_component("TD3", "ALGO")
    db.add_component_version(cid, "hash_v1")
    db.add_component_version(cid, "hash_v2")
    row = db.conn.execute(
        "SELECT MAX(version_number) FROM ComponentVersions WHERE component_id = ?", (cid,)
    ).fetchone()
    assert row[0] == 2


def test_version_numbers_are_per_component(db: DatabaseManager) -> None:
    """version_number restarts from 1 for each component."""
    cid1 = db.add_component("Algo1", "ALGO")
    cid2 = db.add_component("Env1", "ENV")
    db.add_component_version(cid1, "h1")
    db.add_component_version(cid1, "h2")
    vid2 = db.add_component_version(cid2, "h3")
    row = db.conn.execute("SELECT version_number FROM ComponentVersions WHERE id = ?", (vid2,)).fetchone()
    assert row[0] == 1, "Second component's first version should be 1, not 3"


# ---------------------------------------------------------------------------
# HyperparamConfigs — content-addressed, vmap-zone metadata
# ---------------------------------------------------------------------------


def test_add_hyperparam_config_returns_id(db: DatabaseManager) -> None:
    hid = db.add_hyperparam_config({"lr": 1e-3, "gamma": 0.99})
    assert isinstance(hid, int) and hid > 0


def test_add_hyperparam_config_deduplicates(db: DatabaseManager) -> None:
    """Same params must return the same id (content-addressed)."""
    params = {"lr": 3e-4, "gamma": 0.99, "seed": 42}
    id1 = db.add_hyperparam_config(params)
    id2 = db.add_hyperparam_config(params)
    assert id1 == id2


def test_add_run_with_vmap_zone(db: DatabaseManager) -> None:
    exp_id = db.add_experiment("zoned")
    hid = db.add_hyperparam_config({"lr": 1e-3, "arch": "mlp"})
    vmap_zone = {"static_keys": ["arch", "env_name"], "dynamic_keys": ["lr", "seed"]}
    db.add_run(exp_id, None, None, hid, seed=0, vmap_zone=vmap_zone)
    [run] = db.list_runs(exp_id)
    assert run.vmap_zone == vmap_zone


def test_hyperparam_config_hash_is_canonical(db: DatabaseManager) -> None:
    """Key ordering must not affect identity (JSON is sorted)."""
    id1 = db.add_hyperparam_config({"lr": 1e-3, "gamma": 0.99})
    id2 = db.add_hyperparam_config({"gamma": 0.99, "lr": 1e-3})
    assert id1 == id2


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def test_add_experiment_returns_id(db: DatabaseManager) -> None:
    eid = db.add_experiment("PG Ablations", "Testing entropy coefficient sweep")
    assert isinstance(eid, int) and eid > 0


def test_get_experiment_round_trip(db: DatabaseManager) -> None:
    db.add_experiment("ValueFn Study", "Baseline comparison")
    exp = db.get_experiment("ValueFn Study")
    assert exp is not None
    assert exp.name == "ValueFn Study"
    assert exp.description == "Baseline comparison"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class Populated(NamedTuple):
    """A seeded database together with the ids that were seeded into it."""

    db: DatabaseManager
    exp_id: int
    algo_ver_id: int
    env_ver_id: int
    hyper_id: int


@pytest.fixture()
def populated_db(db: DatabaseManager) -> Populated:
    """Return a DatabaseManager with one component/version/config/experiment."""
    algo_id = db.add_component("PPO3", "ALGO")
    env_id = db.add_component("CartPole2", "ENV")
    algo_ver_id = db.add_component_version(algo_id, "ahash")
    env_ver_id = db.add_component_version(env_id, "ehash")
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.add_experiment("Test Exp")
    return Populated(db, exp_id, algo_ver_id, env_ver_id, hyper_id)


def test_add_run_returns_id(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=0)
    assert isinstance(run_id, int) and run_id > 0


def test_add_run_duplicate_raises(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=1, ablation="base")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=1, ablation="base")


def test_list_runs_returns_all_runs_for_experiment(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=0)
    db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=1)

    runs = db.list_runs(exp_id)

    assert [run.seed for run in runs] == [0, 1]


def test_list_runs_without_completed_execution_excludes_completed_execution(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run0 = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=0)
    run1 = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=1)

    execution_id = db.add_execution(hostname="node01")
    db.link_execution_run(execution_id, run0)
    db.update_execution_status(execution_id, "COMPLETED")

    unsatisfied_runs = db.list_runs_without_completed_execution(exp_id)

    assert [run.id for run in unsatisfied_runs] == [run1]


def test_list_runs_without_completed_execution_keeps_failed_only_runs(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=2)

    execution_id = db.add_execution(hostname="node02")
    db.link_execution_run(execution_id, run_id)
    db.update_execution_status(execution_id, "FAILED")

    unsatisfied_runs = db.list_runs_without_completed_execution(exp_id)

    assert [run.id for run in unsatisfied_runs] == [run_id]


# ---------------------------------------------------------------------------
# Executions and ExecutionRuns
# ---------------------------------------------------------------------------


def test_add_execution_defaults_to_pending(db: DatabaseManager) -> None:
    eid = db.add_execution(hostname="node01", git_commit="abc123")
    row = db.conn.execute("SELECT status FROM Executions WHERE id = ?", (eid,)).fetchone()
    assert row[0] == "PENDING"


def test_update_execution_status(db: DatabaseManager) -> None:
    eid = db.add_execution()
    db.update_execution_status(eid, "RUNNING", start_time="2025-01-01T00:00:00Z")
    row = db.conn.execute("SELECT status, start_time FROM Executions WHERE id = ?", (eid,)).fetchone()
    assert row[0] == "RUNNING"
    assert row[1] == "2025-01-01T00:00:00Z"


def test_update_execution_status_rejects_illegal_terminal_rewind(db: DatabaseManager) -> None:
    """
    A terminal execution cannot return to an active state or finish twice.
    """
    eid = db.add_execution()
    db.update_execution_status(eid, "RUNNING")
    db.update_execution_status(eid, "COMPLETED")

    with pytest.raises(ValueError, match="Illegal execution transition"):
        db.update_execution_status(eid, "RUNNING")
    with pytest.raises(ValueError, match="Illegal execution transition"):
        db.update_execution_status(eid, "COMPLETED")


def test_update_execution_status_allows_one_claim(tmp_path: Path) -> None:
    """
    Separate database connections cannot both claim one pending execution.
    """
    db_path = tmp_path / "claim.sqlite"
    with DatabaseManager(db_path) as first:
        first.initialize()
        execution_id = first.add_execution()

    with DatabaseManager(db_path) as first, DatabaseManager(db_path) as second:
        first.initialize()
        second.initialize()
        first.update_execution_status(execution_id, "RUNNING")
        with pytest.raises(ValueError, match="Illegal execution transition"):
            second.update_execution_status(execution_id, "RUNNING")


def test_update_execution_invalid_status_raises(db: DatabaseManager) -> None:
    eid = db.add_execution()
    with pytest.raises(ValueError, match="Invalid status"):
        db.update_execution_status(eid, "LAUNCHED")


def test_add_execution_with_jax_config(db: DatabaseManager) -> None:
    jax_cfg = {"jax_version": "0.4.30", "platform": "gpu", "devices": 4}
    eid = db.add_execution(jax_config=jax_cfg)
    row = db.conn.execute("SELECT jax_config_json FROM Executions WHERE id = ?", (eid,)).fetchone()
    assert row[0] is not None
    parsed = json.loads(row[0])
    assert parsed["platform"] == "gpu"
    assert parsed["devices"] == 4


def test_link_execution_run(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=7)
    exec_id = db.add_execution(hostname="node02")
    db.link_execution_run(exec_id, run_id)
    row = db.conn.execute(
        "SELECT execution_id, run_id FROM ExecutionRuns WHERE execution_id = ? AND run_id = ?",
        (exec_id, run_id),
    ).fetchone()
    assert row is not None


def test_get_execution_round_trip(db: DatabaseManager) -> None:
    execution_id = db.add_execution(hostname="node03", git_commit="deadbeef")

    execution = db.get_execution(execution_id)

    assert execution is not None
    assert execution.hostname == "node03"
    assert execution.git_commit == "deadbeef"


def test_record_execution_artifacts_round_trip(db: DatabaseManager) -> None:
    execution_id = db.add_execution(hostname="node-artifact")

    db.record_execution_artifacts(
        execution_id,
        "/tmp/executions/1",
        manifest_path="/tmp/executions/1/manifest.json",
        metadata={"cohort": "seed-0-3"},
    )

    artifact = db.get_execution_artifacts(execution_id)

    assert artifact is not None
    assert artifact.root_path == "/tmp/executions/1"
    assert artifact.manifest_path == "/tmp/executions/1/manifest.json"
    assert artifact.metadata_json == '{"cohort": "seed-0-3"}'


def test_record_execution_artifacts_updates_existing_row(db: DatabaseManager) -> None:
    """
    Updating runtime artifact data retains metadata recorded during planning.
    """
    execution_id = db.add_execution(hostname="node-artifact-update")
    db.record_execution_artifacts(
        execution_id,
        "/tmp/executions/old",
        metadata={"run_ids": [1], "static_config_json": "{}"},
    )

    db.record_execution_artifacts(
        execution_id,
        "/tmp/executions/new",
        manifest_path="/tmp/executions/new/manifest.json",
        metadata={"trained": True},
    )

    artifact = db.get_execution_artifacts(execution_id)

    assert artifact is not None
    assert artifact.root_path == "/tmp/executions/new"
    assert artifact.manifest_path == "/tmp/executions/new/manifest.json"
    assert json.loads(artifact.metadata_json or "{}") == {
        "run_ids": [1],
        "static_config_json": "{}",
        "trained": True,
    }


def test_one_execution_covers_multiple_runs(populated_db: Populated) -> None:
    """Verify the vmap-zone many-to-one semantic: 1 execution → N runs."""
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    exec_id = db.add_execution(hostname="gpu_node")
    for seed in range(5):
        run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=seed)
        db.link_execution_run(exec_id, run_id)
    count = db.conn.execute(
        "SELECT COUNT(*) FROM ExecutionRuns WHERE execution_id = ?", (exec_id,)
    ).fetchone()[0]
    assert count == 5


def test_canonical_batch_planner_rolls_back_failed_link(populated_db: Populated) -> None:
    """A link failure leaves no partially inserted execution behind."""
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=0)
    db.conn.execute(
        "CREATE TRIGGER fail_execution_link BEFORE INSERT ON ExecutionRuns "
        "BEGIN SELECT RAISE(ABORT, 'forced link failure'); END;",
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced link failure"):
        db.plan_experiment_execution_batches(exp_id, "/tmp/executions")

    assert db.list_executions() == []


def test_canonical_batch_planner_skips_claimed_runs(populated_db: Populated) -> None:
    """Completed or pending claims are excluded from later canonical planning."""
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    completed_run = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=0)
    fresh_run = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=1)
    execution_id = db.add_execution()
    db.link_execution_run(execution_id, completed_run)
    db.update_execution_status(execution_id, "COMPLETED")

    planned = db.plan_experiment_execution_batches(exp_id, "/tmp/executions")

    assert len(planned) == 1
    assert [run.id for run in db.list_execution_runs(planned[0].execution_id)] == [fresh_run]


def test_get_latest_completed_execution_for_run_returns_latest(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=9)

    older_execution = db.add_execution(hostname="node-old")
    db.link_execution_run(older_execution, run_id)
    db.update_execution_status(older_execution, "COMPLETED", end_time="2026-03-30T00:00:00Z")

    newer_execution = db.add_execution(hostname="node-new")
    db.link_execution_run(newer_execution, run_id)
    db.update_execution_status(newer_execution, "COMPLETED", end_time="2026-03-30T00:05:00Z")

    latest = db.get_latest_completed_execution_for_run(run_id)

    assert latest is not None
    assert latest.id == newer_execution


def test_list_run_execution_statuses_reports_every_attempt(populated_db: Populated) -> None:
    """Each run maps to all its execution statuses, and a never-executed run maps to none."""
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    retried_run = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=21)
    unexecuted_run = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=22)
    for status in ("FAILED", "COMPLETED"):
        execution = db.add_execution()
        db.link_execution_run(execution, retried_run)
        db.update_execution_status(execution, status)

    statuses = db.list_run_execution_statuses(exp_id)

    assert statuses[retried_run] == frozenset({"FAILED", "COMPLETED"})
    assert statuses[unexecuted_run] == frozenset()


def test_list_unsatisfied_run_batches_groups_by_static_vmap_zone(db: DatabaseManager) -> None:
    algo_id = db.add_component("BatchAlgo", "ALGO")
    env_id = db.add_component("BatchEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Batch Experiment")

    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": ["lr"]}
    shared_static_h1 = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3})
    shared_static_h2 = db.add_hyperparam_config({"arch": "mlp", "lr": 3e-4})
    different_static = db.add_hyperparam_config({"arch": "cnn", "lr": 1e-3})

    db.add_run(exp_id, algo_ver, env_ver, shared_static_h1, seed=0, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, shared_static_h2, seed=1, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, different_static, seed=2, vmap_zone=vmap_zone)

    batches = db.list_unsatisfied_run_batches(exp_id)

    assert len(batches) == 2
    assert [len(batch.run_ids) for batch in batches] == [1, 2]
    assert batches[0].static_config == {"arch": "cnn"}
    assert batches[1].static_config == {"arch": "mlp"}


def _selector_fixture(db: DatabaseManager) -> int:
    algo_id = db.add_component("SelectAlgo", "ALGO")
    env_id = db.add_component("SelectEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Selector Experiment")

    vmap_zone = {"static_keys": ["algorithm"], "dynamic_keys": []}
    for algorithm in ("sac", "gi_sac"):
        hyper_id = db.add_hyperparam_config({"algorithm": algorithm, "lr": 1e-3})
        for seed in range(4):
            db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=seed, vmap_zone=vmap_zone)
    return exp_id


def test_the_selector_keeps_only_runs_matching_every_clause(db: DatabaseManager) -> None:
    """Clauses AND together across keys, so two clauses narrow further than either alone."""
    exp_id = _selector_fixture(db)

    both = db.list_unsatisfied_run_batches(exp_id, only={"algorithm": "gi_sac", "seed": "0"})
    algorithm_only = db.list_unsatisfied_run_batches(exp_id, only={"algorithm": "gi_sac"})

    assert [len(batch.run_ids) for batch in both] == [1]
    assert [len(batch.run_ids) for batch in algorithm_only] == [4]


def test_the_selector_ors_comma_separated_values_within_one_clause(db: DatabaseManager) -> None:
    """A seed range is expressible, which repeated strict-equality clauses could not do.

    Repeating a key would overwrite it in the parsed mapping rather than widen it, so the
    comma is what makes set membership reachable at all.
    """
    exp_id = _selector_fixture(db)

    batches = db.list_unsatisfied_run_batches(exp_id, only={"algorithm": "gi_sac", "seed": "0,1,2"})

    assert [len(batch.run_ids) for batch in batches] == [3]


def test_the_selector_narrows_a_partial_batch_rather_than_skipping_it(db: DatabaseManager) -> None:
    """A batch is a vmap zone, so selecting some of its runs must still plan those runs.

    ``find_executions_by_hyperparams`` skips partially matching batches, because
    under-selecting is the safe failure when invalidating results. Here the failure
    inverts: skipping would make a single-seed selector execute nothing at all.
    """
    exp_id = _selector_fixture(db)

    unfiltered = db.list_unsatisfied_run_batches(exp_id)
    narrowed = db.list_unsatisfied_run_batches(exp_id, only={"seed": "0"})

    assert [len(batch.run_ids) for batch in unfiltered] == [4, 4]
    assert [len(batch.run_ids) for batch in narrowed] == [1, 1]


def test_the_selector_matches_a_seed_that_is_absent_from_the_config_blob(db: DatabaseManager) -> None:
    """``seed`` is a Runs column, not a hyperparameter, so it needs the column to be reachable."""
    exp_id = _selector_fixture(db)

    batches = db.list_unsatisfied_run_batches(exp_id, only={"seed": "3"})

    assert sum(len(batch.run_ids) for batch in batches) == 2


def test_the_selector_selects_nothing_for_an_unknown_key(db: DatabaseManager) -> None:
    """An unmatched key must not silently widen the selection to everything."""
    exp_id = _selector_fixture(db)

    assert db.list_unsatisfied_run_batches(exp_id, only={"nonexistent": "x"}) == []


def test_the_selector_compares_values_as_strings_like_the_where_dialect(db: DatabaseManager) -> None:
    """``1e-3`` does not match a stored ``0.001``, matching find_executions_by_hyperparams.

    Pinned so the two dialects cannot drift apart silently; a caller who learns the rule
    from one flag can rely on it in the other.
    """
    exp_id = _selector_fixture(db)

    assert db.list_unsatisfied_run_batches(exp_id, only={"lr": "1e-3"}) == []
    assert sum(len(b.run_ids) for b in db.list_unsatisfied_run_batches(exp_id, only={"lr": "0.001"})) == 8


def test_list_unsatisfied_run_batches_respects_batch_limit(db: DatabaseManager) -> None:
    algo_id = db.add_component("ChunkAlgo", "ALGO")
    env_id = db.add_component("ChunkEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Chunk Experiment")

    hyper_id = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3})
    for seed in range(5):
        db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=seed, vmap_zone={"static_keys": ["arch"]})

    batches = db.list_unsatisfied_run_batches(exp_id, max_runs_per_batch=2)

    assert [len(batch.run_ids) for batch in batches] == [2, 2, 1]


def test_list_unsatisfied_run_batches_chunks_do_not_interleave_hyper_configs(db: DatabaseManager) -> None:
    """Chunking must group by hyperparameter configuration, not interleave across it.

    Two hyperparameter configs share the same static config (same ``arch``, differing
    only in the vmapped ``lr``), so they land in the same batch bucket. Runs for both
    configs are added interleaved by seed. With a batch limit smaller than the total
    run count, each chunk must still contain runs from only one hyperparameter config --
    a seed-first SQL ordering would instead interleave the two configs' seeds across
    chunk boundaries.
    """
    algo_id = db.add_component("InterleaveAlgo", "ALGO")
    env_id = db.add_component("InterleaveEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Interleave Experiment")

    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": ["lr"]}
    hyper_a = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3})
    hyper_b = db.add_hyperparam_config({"arch": "mlp", "lr": 2e-3})

    hyper_a_run_ids = []
    hyper_b_run_ids = []
    for seed in range(3):
        hyper_a_run_ids.append(db.add_run(exp_id, algo_ver, env_ver, hyper_a, seed=seed, vmap_zone=vmap_zone))
        hyper_b_run_ids.append(db.add_run(exp_id, algo_ver, env_ver, hyper_b, seed=seed, vmap_zone=vmap_zone))

    batches = db.list_unsatisfied_run_batches(exp_id, max_runs_per_batch=3)

    assert len(batches) == 2
    assert {run_id for run_id in batches[0].run_ids} == set(hyper_a_run_ids)
    assert {run_id for run_id in batches[1].run_ids} == set(hyper_b_run_ids)


def test_plan_experiment_execution_batches_creates_one_execution_per_batch(db: DatabaseManager) -> None:
    algo_id = db.add_component("PlannerAlgo", "ALGO")
    env_id = db.add_component("PlannerEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Planner Experiment")

    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": ["lr"]}
    hyper_a = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3})
    hyper_b = db.add_hyperparam_config({"arch": "cnn", "lr": 1e-3})
    db.add_run(exp_id, algo_ver, env_ver, hyper_a, seed=0, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, hyper_a, seed=1, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, hyper_b, seed=2, vmap_zone=vmap_zone)

    execution_ids = [
        planned.execution_id
        for planned in db.plan_experiment_execution_batches(
            exp_id, "/tmp/executions", hostname="batch-planner",
        )
    ]

    assert len(execution_ids) == 2
    first_batch_runs = db.list_execution_runs(execution_ids[0])
    second_batch_runs = db.list_execution_runs(execution_ids[1])
    assert [len(first_batch_runs), len(second_batch_runs)] == [1, 2]
    assert all(_require(db.get_execution(execution_id)).hostname == "batch-planner" for execution_id in execution_ids)


def test_concurrent_batch_planning_has_one_winner(tmp_path: Path) -> None:
    """
    Two database connections planning the same experiment create one batch.
    """
    db_path = tmp_path / "concurrent.sqlite"
    with DatabaseManager(db_path) as database:
        database.initialize()
        algo_id = database.add_component("ConcurrentAlgo", "ALGO")
        env_id = database.add_component("ConcurrentEnv", "ENV")
        algo_ver = database.add_component_version(algo_id, "algo-hash")
        env_ver = database.add_component_version(env_id, "env-hash")
        exp_id = database.add_experiment("Concurrent Experiment")
        hyper_id = database.add_hyperparam_config({"lr": 1e-3})
        database.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=0)

    def plan() -> list[int]:
        with DatabaseManager(db_path) as database:
            database.initialize()
            return [
                planned.execution_id
                for planned in database.plan_experiment_execution_batches(
                    exp_id, tmp_path / "executions", hostname="planner",
                )
            ]

    def run_plan(_: int):
        return plan()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_plan, range(2)))

    assert sorted(len(result) for result in results) == [0, 1]
    with DatabaseManager(db_path) as database:
        database.initialize()
        assert len(database.list_executions(exp_id)) == 1


def test_plan_experiment_execution_batches_records_artifact_paths(db: DatabaseManager) -> None:
    """
    Replanning before execution does not create another active batch.
    """
    algo_id = db.add_component("FacadeAlgo", "ALGO")
    env_id = db.add_component("FacadeEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Facade Experiment")

    hyper_id = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3})
    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": ["lr"]}
    db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=0, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=1, vmap_zone=vmap_zone)

    planned = db.plan_experiment_execution_batches(
        exp_id,
        "/tmp/experiment-executions",
        hostname="facade-node",
    )

    assert len(planned) == 1
    assert planned[0].root_path.endswith(f"/{planned[0].execution_id}")
    assert planned[0].manifest_path.endswith(f"/{planned[0].execution_id}/manifest.json")

    artifacts = db.get_execution_artifacts(planned[0].execution_id)
    assert artifacts is not None
    assert artifacts.root_path == planned[0].root_path
    assert artifacts.manifest_path == planned[0].manifest_path
    assert '"run_ids": [1, 2]' in (artifacts.metadata_json or "")

    assert db.plan_experiment_execution_batches(exp_id, "/tmp/experiment-executions") == []


def test_plan_experiment_execution_batches_rolls_back_all_batches_on_artifact_failure(
    db: DatabaseManager,
) -> None:
    """A later artifact failure must roll back every execution from the batch plan."""
    algo_id = db.add_component("AtomicPlannerAlgo", "ALGO")
    env_id = db.add_component("AtomicPlannerEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Atomic Planner Experiment")
    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": []}
    mlp_hyper_id = db.add_hyperparam_config({"arch": "mlp"})
    cnn_hyper_id = db.add_hyperparam_config({"arch": "cnn"})
    db.add_run(exp_id, algo_ver, env_ver, mlp_hyper_id, seed=0, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, cnn_hyper_id, seed=1, vmap_zone=vmap_zone)
    db.conn.execute(
        "CREATE TRIGGER fail_second_artifact_insert "
        "BEFORE INSERT ON ExecutionArtifacts "
        "WHEN (SELECT COUNT(*) FROM ExecutionArtifacts) >= 1 "
        "BEGIN SELECT RAISE(ABORT, 'forced artifact failure'); END;",
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced artifact failure"):
        db.plan_experiment_execution_batches(exp_id, "/tmp/atomic-experiment-executions")

    assert db.list_executions() == []
    artifact_count = db.conn.execute("SELECT COUNT(*) FROM ExecutionArtifacts").fetchone()[0]
    assert artifact_count == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_opens_and_closes() -> None:
    with DatabaseManager(":memory:") as db:
        db.initialize()
        assert db.conn is not None
    # After __exit__, conn should be None
    assert db._conn is None  # noqa: SLF001


def test_reconnect_uses_the_current_connection(tmp_path: Path) -> None:
    """Repository calls after reconnecting must use the replacement SQLite connection."""
    db_path = tmp_path / "reconnect.sqlite"
    database = DatabaseManager(db_path)
    database.connect()
    database.initialize()
    database.add_experiment("Reconnect Experiment")
    database.close()

    database.connect()
    database.initialize()

    assert database.get_experiment("Reconnect Experiment") is not None
    database.close()


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_invalidate_execution_sets_status(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=20)

    execution_id = db.add_execution(hostname="node-inv")
    db.link_execution_run(execution_id, run_id)
    db.update_execution_status(execution_id, "COMPLETED")

    result = db.invalidate_execution(execution_id)

    assert result is True
    assert _require(db.get_execution(execution_id)).status == "INVALID"


def test_invalidate_execution_nonexistent_returns_false(db: DatabaseManager) -> None:
    assert db.invalidate_execution(999999) is False


def test_invalidate_execution_makes_runs_unsatisfied(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=21)

    execution_id = db.add_execution(hostname="node-inv2")
    db.link_execution_run(execution_id, run_id)
    db.update_execution_status(execution_id, "COMPLETED")

    assert run_id not in [r.id for r in db.list_runs_without_completed_execution(exp_id)]

    db.invalidate_execution(execution_id)

    assert run_id in [r.id for r in db.list_runs_without_completed_execution(exp_id)]


def test_invalidate_executions_by_commit(db: DatabaseManager) -> None:
    e1 = db.add_execution(hostname="n1", git_commit="aaa111")
    db.update_execution_status(e1, "COMPLETED")
    e2 = db.add_execution(hostname="n2", git_commit="aaa111")
    db.update_execution_status(e2, "COMPLETED")
    e3 = db.add_execution(hostname="n3", git_commit="bbb222")
    db.update_execution_status(e3, "COMPLETED")

    invalidated = db.invalidate_executions_by_commit("aaa111")

    assert sorted(invalidated) == sorted([e1, e2])
    assert _require(db.get_execution(e1)).status == "INVALID"
    assert _require(db.get_execution(e2)).status == "INVALID"
    assert _require(db.get_execution(e3)).status == "COMPLETED"


def test_find_executions_by_hyperparams_matches_full_batch(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    matching_hyper_id = db.add_hyperparam_config({"lr": 1e-3, "algorithm": "gi_qrc"})
    other_hyper_id = db.add_hyperparam_config({"lr": 1e-3, "algorithm": "baseline"})

    matching_run = db.add_run(exp_id, algo_ver_id, env_ver_id, matching_hyper_id, seed=0)
    other_run = db.add_run(exp_id, algo_ver_id, env_ver_id, other_hyper_id, seed=0)

    matching_execution = db.add_execution(hostname="n1")
    db.link_execution_run(matching_execution, matching_run)
    db.update_execution_status(matching_execution, "COMPLETED")

    other_execution = db.add_execution(hostname="n2")
    db.link_execution_run(other_execution, other_run)
    db.update_execution_status(other_execution, "COMPLETED")

    matched = db.find_executions_by_hyperparams(exp_id, {"algorithm": "gi_qrc"})

    assert [row.id for row in matched] == [matching_execution]


def test_find_executions_by_hyperparams_skips_partial_batch(populated_db: Populated) -> None:
    """
    An execution batch is skipped, not invalidated, when only some of its
    covered runs match -- silently invalidating unselected runs is the
    failure this lookup exists to prevent.
    """
    db, exp_id, algo_ver_id, env_ver_id, _hyper_id = populated_db
    matching_hyper_id = db.add_hyperparam_config({"algorithm": "gi_qrc"})
    other_hyper_id = db.add_hyperparam_config({"algorithm": "baseline"})

    matching_run = db.add_run(exp_id, algo_ver_id, env_ver_id, matching_hyper_id, seed=0)
    other_run = db.add_run(exp_id, algo_ver_id, env_ver_id, other_hyper_id, seed=1)

    mixed_execution = db.add_execution(hostname="n1")
    db.link_execution_run(mixed_execution, matching_run)
    db.link_execution_run(mixed_execution, other_run)
    db.update_execution_status(mixed_execution, "COMPLETED")

    matched = db.find_executions_by_hyperparams(exp_id, {"algorithm": "gi_qrc"})

    assert matched == []


def test_find_executions_by_hyperparams_compares_json_value_str(populated_db: Populated) -> None:
    db, exp_id, algo_ver_id, env_ver_id, _hyper_id = populated_db
    hyper_id = db.add_hyperparam_config({"learning_rate": 0.001})
    run_id = db.add_run(exp_id, algo_ver_id, env_ver_id, hyper_id, seed=0)
    execution_id = db.add_execution(hostname="n1")
    db.link_execution_run(execution_id, run_id)
    db.update_execution_status(execution_id, "COMPLETED")

    assert [row.id for row in db.find_executions_by_hyperparams(exp_id, {"learning_rate": "0.001"})] == [execution_id]
    assert db.find_executions_by_hyperparams(exp_id, {"learning_rate": "1e-3"}) == []


def test_invalidate_by_commit_skips_pending_and_running(db: DatabaseManager) -> None:
    e_pending = db.add_execution(hostname="n1", git_commit="ccc333")
    e_running = db.add_execution(hostname="n2", git_commit="ccc333")
    db.update_execution_status(e_running, "RUNNING")
    e_completed = db.add_execution(hostname="n3", git_commit="ccc333")
    db.update_execution_status(e_completed, "COMPLETED")

    invalidated = db.invalidate_executions_by_commit("ccc333")

    assert invalidated == [e_completed]
    assert _require(db.get_execution(e_pending)).status == "PENDING"
    assert _require(db.get_execution(e_running)).status == "RUNNING"
    assert _require(db.get_execution(e_completed)).status == "INVALID"


# ---------------------------------------------------------------------------
# list_executions
# ---------------------------------------------------------------------------


def test_list_executions_unfiltered(db: DatabaseManager) -> None:
    db.add_execution(hostname="n1")
    db.add_execution(hostname="n2")

    rows = db.list_executions()

    assert len(rows) == 2


def test_list_executions_filter_by_status(db: DatabaseManager) -> None:
    _ = db.add_execution(hostname="n1")
    e2 = db.add_execution(hostname="n2")
    db.update_execution_status(e2, "COMPLETED")

    rows = db.list_executions(status="COMPLETED")

    assert len(rows) == 1
    assert rows[0].id == e2


def test_list_executions_filter_by_experiment(populated_db: Populated) -> None:
    db, exp1_id, algo_ver_id, env_ver_id, hyper_id = populated_db
    exp2_id = db.add_experiment("Other Exp")

    run1 = db.add_run(exp1_id, algo_ver_id, env_ver_id, hyper_id, seed=30)
    run2 = db.add_run(exp2_id, algo_ver_id, env_ver_id, hyper_id, seed=31)

    exec1 = db.add_execution(hostname="n1")
    db.link_execution_run(exec1, run1)
    exec2 = db.add_execution(hostname="n2")
    db.link_execution_run(exec2, run2)

    rows = db.list_executions(exp1_id)

    assert len(rows) == 1
    assert rows[0].id == exec1


def test_list_executions_filter_by_git_commit(db: DatabaseManager) -> None:
    db.add_execution(hostname="n1", git_commit="ccc333")
    db.add_execution(hostname="n2", git_commit="ddd444")

    rows = db.list_executions(git_commit="ccc333")

    assert len(rows) == 1
    assert rows[0].git_commit == "ccc333"
