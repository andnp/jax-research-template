"""Medium (< 1 s) integration tests for the experiment-definition DatabaseManager.

These tests exercise the actual SQLite interactions using an in-memory database,
verifying schema creation, CRUD operations, auto-versioning, and the "latest"
version pointer logic.
"""

import json
import sqlite3

import pytest
from experiment_definition.db import DatabaseManager

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
    required = {"Components", "ComponentVersions", "HyperparamConfigs", "Experiments", "Runs", "Executions", "ExecutionRuns", "ParameterSpecs", "MetricSpecs"}
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


def test_get_latest_version_returns_highest(db: DatabaseManager) -> None:
    cid = db.add_component("PPO2", "ALGO")
    db.add_component_version(cid, "hash_a", notes="initial")
    db.add_component_version(cid, "hash_b", notes="bug fix")
    db.add_component_version(cid, "hash_c", notes="optimisation")
    latest = db.get_latest_version(cid)
    assert latest is not None
    assert latest.version_number == 3
    assert latest.notes == "optimisation"


def test_get_latest_version_no_versions_returns_none(db: DatabaseManager) -> None:
    cid = db.add_component("EmptyAlgo", "ALGO")
    assert db.get_latest_version(cid) is None


def test_hash_changed_true_when_no_versions(db: DatabaseManager) -> None:
    cid = db.add_component("NewAlgo", "ALGO")
    assert db.hash_changed(cid, "any_hash") is True


def test_hash_changed_false_when_same(db: DatabaseManager) -> None:
    cid = db.add_component("StableAlgo", "ALGO")
    db.add_component_version(cid, "stable_hash")
    assert db.hash_changed(cid, "stable_hash") is False


def test_hash_changed_true_when_different(db: DatabaseManager) -> None:
    cid = db.add_component("ChangedAlgo", "ALGO")
    db.add_component_version(cid, "old_hash")
    assert db.hash_changed(cid, "new_hash") is True


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


def test_add_hyperparam_config_with_vmap_zone(db: DatabaseManager) -> None:
    vmap_zone = {"static_keys": ["arch", "env_name"], "dynamic_keys": ["lr", "seed"]}
    hid = db.add_hyperparam_config({"lr": 1e-3, "arch": "mlp"}, vmap_zone=vmap_zone)
    row = db.get_hyperparam_config(hid)
    assert row is not None
    assert row.vmap_zone_json is not None
    parsed = json.loads(row.vmap_zone_json)
    assert parsed["static_keys"] == ["arch", "env_name"]
    assert parsed["dynamic_keys"] == ["lr", "seed"]


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


@pytest.fixture()
def populated_db(db: DatabaseManager) -> DatabaseManager:
    """Return a DatabaseManager with one component/version/config/experiment."""
    algo_id = db.add_component("PPO3", "ALGO")
    env_id = db.add_component("CartPole2", "ENV")
    db.add_component_version(algo_id, "ahash")
    db.add_component_version(env_id, "ehash")
    db.add_hyperparam_config({"lr": 1e-3})
    db.add_experiment("Test Exp")
    return db


def test_add_run_returns_id(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=0)  # type: ignore[union-attr]
    assert isinstance(run_id, int) and run_id > 0


def test_add_run_duplicate_raises(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=1, ablation="base")  # type: ignore[union-attr]
    with pytest.raises(sqlite3.IntegrityError):
        db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=1, ablation="base")  # type: ignore[union-attr]


def test_ensure_experiment_reuses_existing_id(db: DatabaseManager) -> None:
    first_id = db.ensure_experiment("Declarative Sweep", "first")
    second_id = db.ensure_experiment("Declarative Sweep", "second")

    assert first_id == second_id


def test_list_runs_returns_all_runs_for_experiment(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=0)  # type: ignore[union-attr]
    db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=1)  # type: ignore[union-attr]

    runs = db.list_runs(exp_id)

    assert [run.seed for run in runs] == [0, 1]


def test_list_unsatisfied_runs_excludes_completed_execution(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run0 = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=0)  # type: ignore[union-attr]
    run1 = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=1)  # type: ignore[union-attr]

    execution_id = db.add_execution(hostname="node01")
    db.link_execution_run(execution_id, run0)
    db.update_execution_status(execution_id, "COMPLETED")

    unsatisfied_runs = db.list_unsatisfied_runs(exp_id)

    assert [run.id for run in unsatisfied_runs] == [run1]


def test_list_unsatisfied_runs_keeps_failed_only_runs(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=2)  # type: ignore[union-attr]

    execution_id = db.add_execution(hostname="node02")
    db.link_execution_run(execution_id, run_id)
    db.update_execution_status(execution_id, "FAILED")

    unsatisfied_runs = db.list_unsatisfied_runs(exp_id)

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


def test_link_execution_run(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=7)  # type: ignore[union-attr]
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
    execution_id = db.add_execution(hostname="node-artifact-update")
    db.record_execution_artifacts(execution_id, "/tmp/executions/old")

    db.record_execution_artifacts(
        execution_id,
        "/tmp/executions/new",
        manifest_path="/tmp/executions/new/manifest.json",
    )

    artifact = db.get_execution_artifacts(execution_id)

    assert artifact is not None
    assert artifact.root_path == "/tmp/executions/new"
    assert artifact.manifest_path == "/tmp/executions/new/manifest.json"


def test_one_execution_covers_multiple_runs(populated_db: DatabaseManager) -> None:
    """Verify the vmap-zone many-to-one semantic: 1 execution → N runs."""
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    exec_id = db.add_execution(hostname="gpu_node")
    for seed in range(5):
        run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=seed)  # type: ignore[union-attr]
        db.link_execution_run(exec_id, run_id)
    count = db.conn.execute(
        "SELECT COUNT(*) FROM ExecutionRuns WHERE execution_id = ?", (exec_id,)
    ).fetchone()[0]
    assert count == 5


def test_plan_execution_creates_pending_execution_and_links_runs(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_ids = [
        db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=seed)  # type: ignore[union-attr]
        for seed in (0, 1)
    ]

    execution_id = db.plan_execution(run_ids, hostname="planner-node")

    execution = db.get_execution(execution_id)
    linked_runs = db.list_execution_runs(execution_id)

    assert execution is not None
    assert execution.status == "PENDING"
    assert execution.hostname == "planner-node"
    assert [run.id for run in linked_runs] == run_ids


def test_plan_execution_requires_non_empty_run_ids(db: DatabaseManager) -> None:
    with pytest.raises(ValueError, match="run_ids must not be empty"):
        db.plan_execution([])


def test_plan_unsatisfied_execution_selects_only_unsatisfied_runs(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    completed_run = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=0)  # type: ignore[union-attr]
    failed_run = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=1)  # type: ignore[union-attr]
    fresh_run = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=2)  # type: ignore[union-attr]

    completed_execution = db.add_execution(hostname="node-complete")
    db.link_execution_run(completed_execution, completed_run)
    db.update_execution_status(completed_execution, "COMPLETED", end_time="2026-03-30T00:00:00Z")

    failed_execution = db.add_execution(hostname="node-fail")
    db.link_execution_run(failed_execution, failed_run)
    db.update_execution_status(failed_execution, "FAILED", end_time="2026-03-30T00:01:00Z")

    planned_execution = db.plan_unsatisfied_execution(exp_id, hostname="planner")

    assert planned_execution is not None
    linked_runs = db.list_execution_runs(planned_execution)
    assert [run.id for run in linked_runs] == [failed_run, fresh_run]


def test_plan_unsatisfied_execution_respects_limit(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    for seed in range(3):
        db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=seed)  # type: ignore[union-attr]

    planned_execution = db.plan_unsatisfied_execution(exp_id, limit=2)

    assert planned_execution is not None
    linked_runs = db.list_execution_runs(planned_execution)
    assert len(linked_runs) == 2
    assert [run.seed for run in linked_runs] == [0, 1]


def test_plan_unsatisfied_execution_returns_none_when_all_runs_satisfied(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=0)  # type: ignore[union-attr]

    execution_id = db.add_execution(hostname="node04")
    db.link_execution_run(execution_id, run_id)
    db.update_execution_status(execution_id, "COMPLETED")

    assert db.plan_unsatisfied_execution(exp_id) is None


def test_get_latest_completed_execution_for_run_returns_latest(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=9)  # type: ignore[union-attr]

    older_execution = db.add_execution(hostname="node-old")
    db.link_execution_run(older_execution, run_id)
    db.update_execution_status(older_execution, "COMPLETED", end_time="2026-03-30T00:00:00Z")

    newer_execution = db.add_execution(hostname="node-new")
    db.link_execution_run(newer_execution, run_id)
    db.update_execution_status(newer_execution, "COMPLETED", end_time="2026-03-30T00:05:00Z")

    latest = db.get_latest_completed_execution_for_run(run_id)

    assert latest is not None
    assert latest.id == newer_execution


def test_get_latest_completed_artifacts_for_run_returns_artifacts_for_latest_execution(populated_db: DatabaseManager) -> None:
    db = populated_db
    algo_ver = db.get_latest_version(db.get_component("PPO3").id)  # type: ignore[union-attr]
    env_ver = db.get_latest_version(db.get_component("CartPole2").id)  # type: ignore[union-attr]
    hyper_id = db.add_hyperparam_config({"lr": 1e-3})
    exp_id = db.get_experiment("Test Exp").id  # type: ignore[union-attr]
    run_id = db.add_run(exp_id, algo_ver.id, env_ver.id, hyper_id, seed=11)  # type: ignore[union-attr]

    older_execution = db.add_execution(hostname="node-old")
    db.link_execution_run(older_execution, run_id)
    db.update_execution_status(older_execution, "COMPLETED", end_time="2026-03-30T00:00:00Z")
    db.record_execution_artifacts(older_execution, "/tmp/executions/old")

    newer_execution = db.add_execution(hostname="node-new")
    db.link_execution_run(newer_execution, run_id)
    db.update_execution_status(newer_execution, "COMPLETED", end_time="2026-03-30T00:05:00Z")
    db.record_execution_artifacts(newer_execution, "/tmp/executions/new")

    artifact = db.get_latest_completed_artifacts_for_run(run_id)

    assert artifact is not None
    assert artifact.execution_id == newer_execution
    assert artifact.root_path == "/tmp/executions/new"


def test_list_unsatisfied_run_batches_groups_by_static_vmap_zone(db: DatabaseManager) -> None:
    algo_id = db.add_component("BatchAlgo", "ALGO")
    env_id = db.add_component("BatchEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Batch Experiment")

    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": ["lr"]}
    shared_static_h1 = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3}, vmap_zone=vmap_zone)
    shared_static_h2 = db.add_hyperparam_config({"arch": "mlp", "lr": 3e-4}, vmap_zone=vmap_zone)
    different_static = db.add_hyperparam_config({"arch": "cnn", "lr": 1e-3}, vmap_zone=vmap_zone)

    db.add_run(exp_id, algo_ver, env_ver, shared_static_h1, seed=0)
    db.add_run(exp_id, algo_ver, env_ver, shared_static_h2, seed=1)
    db.add_run(exp_id, algo_ver, env_ver, different_static, seed=2)

    batches = db.list_unsatisfied_run_batches(exp_id)

    assert len(batches) == 2
    assert [len(batch.run_ids) for batch in batches] == [1, 2]
    assert batches[0].static_config_json == '{"arch": "cnn"}'
    assert batches[1].static_config_json == '{"arch": "mlp"}'


def test_list_unsatisfied_run_batches_respects_batch_limit(db: DatabaseManager) -> None:
    algo_id = db.add_component("ChunkAlgo", "ALGO")
    env_id = db.add_component("ChunkEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Chunk Experiment")

    hyper_id = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3}, vmap_zone={"static_keys": ["arch"]})
    for seed in range(5):
        db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=seed)

    batches = db.list_unsatisfied_run_batches(exp_id, max_runs_per_batch=2)

    assert [len(batch.run_ids) for batch in batches] == [2, 2, 1]


def test_plan_unsatisfied_execution_batches_creates_one_execution_per_batch(db: DatabaseManager) -> None:
    algo_id = db.add_component("PlannerAlgo", "ALGO")
    env_id = db.add_component("PlannerEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Planner Experiment")

    vmap_zone = {"static_keys": ["arch"], "dynamic_keys": ["lr"]}
    hyper_a = db.add_hyperparam_config({"arch": "mlp", "lr": 1e-3}, vmap_zone=vmap_zone)
    hyper_b = db.add_hyperparam_config({"arch": "cnn", "lr": 1e-3}, vmap_zone=vmap_zone)
    db.add_run(exp_id, algo_ver, env_ver, hyper_a, seed=0)
    db.add_run(exp_id, algo_ver, env_ver, hyper_a, seed=1)
    db.add_run(exp_id, algo_ver, env_ver, hyper_b, seed=2)

    execution_ids = db.plan_unsatisfied_execution_batches(exp_id, hostname="batch-planner")

    assert len(execution_ids) == 2
    first_batch_runs = db.list_execution_runs(execution_ids[0])
    second_batch_runs = db.list_execution_runs(execution_ids[1])
    assert [len(first_batch_runs), len(second_batch_runs)] == [1, 2]
    assert all(db.get_execution(execution_id).hostname == "batch-planner" for execution_id in execution_ids)  # type: ignore[union-attr]


def test_plan_experiment_execution_batches_records_artifact_paths(db: DatabaseManager) -> None:
    algo_id = db.add_component("FacadeAlgo", "ALGO")
    env_id = db.add_component("FacadeEnv", "ENV")
    algo_ver = db.add_component_version(algo_id, "algo-hash")
    env_ver = db.add_component_version(env_id, "env-hash")
    exp_id = db.add_experiment("Facade Experiment")

    hyper_id = db.add_hyperparam_config(
        {"arch": "mlp", "lr": 1e-3},
        vmap_zone={"static_keys": ["arch"], "dynamic_keys": ["lr"]},
    )
    db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=0)
    db.add_run(exp_id, algo_ver, env_ver, hyper_id, seed=1)

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


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_opens_and_closes() -> None:
    with DatabaseManager(":memory:") as db:
        db.initialize()
        assert db.conn is not None
    # After __exit__, conn should be None
    assert db._conn is None  # noqa: SLF001
