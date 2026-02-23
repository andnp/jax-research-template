"""Medium (< 1 s) integration tests for the experiment-definition DatabaseManager.

These tests exercise the actual SQLite interactions using an in-memory database,
verifying schema creation, CRUD operations, auto-versioning, and the "latest"
version pointer logic.
"""

import json
import sqlite3

import pytest
from experiment_definition import DatabaseManager

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


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_opens_and_closes() -> None:
    with DatabaseManager(":memory:") as db:
        db.initialize()
        assert db.conn is not None
    # After __exit__, conn should be None
    assert db._conn is None  # noqa: SLF001
