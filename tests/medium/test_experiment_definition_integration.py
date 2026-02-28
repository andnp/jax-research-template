"""Medium integration tests for experiment-definition — must complete in < 1s each."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from experiment_definition.component import Component, ComponentType
from experiment_definition.db import _generate_configs, _hash_dict
from experiment_definition.experiment import Experiment

# ── Config generation ─────────────────────────────────────────────────────────

class TestGenerateConfigs:
    def test_simple_cartesian(self) -> None:
        exp = Experiment("Test")
        exp.add_parameter("lr", [1e-3, 3e-4])
        exp.add_parameter("gamma", [0.99])
        configs = _generate_configs(exp._state)
        assert len(configs) == 2
        assert all("lr" in c and "gamma" in c for c in configs)

    def test_conditional_expanded(self) -> None:
        exp = Experiment("Test")
        exp.add_parameter("use_gae", [True, False])
        with exp.when(use_gae=True):
            exp.add_parameter("gae_lambda", [0.9, 0.95])
        configs = _generate_configs(exp._state)
        # use_gae=False → 1 config; use_gae=True → 2 configs (gae_lambda)
        assert len(configs) == 3
        gae_configs = [c for c in configs if c.get("use_gae") is True]
        assert len(gae_configs) == 2
        assert all("gae_lambda" in c for c in gae_configs)

    def test_no_conditional_key_absent(self) -> None:
        exp = Experiment("Test")
        exp.add_parameter("use_gae", [False])
        with exp.when(use_gae=True):
            exp.add_parameter("gae_lambda", [0.9])
        configs = _generate_configs(exp._state)
        assert len(configs) == 1
        assert "gae_lambda" not in configs[0]

    def test_empty_produces_single_empty_config(self) -> None:
        exp = Experiment("Test")
        configs = _generate_configs(exp._state)
        assert configs == [{}]

    def test_hash_dict_deterministic(self) -> None:
        d = {"b": 2, "a": 1}
        assert _hash_dict(d) == _hash_dict({"a": 1, "b": 2})


# ── SQLite sync ───────────────────────────────────────────────────────────────

class TestSyncToDb:
    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "exp.sqlite"

    def test_creates_file(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.sync(db_path)
        assert db_path.exists()

    def test_experiment_row_inserted(self, db_path: Path) -> None:
        exp = Experiment("My Sweep", description="hello")
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            row = con.execute("SELECT name, description FROM Experiments").fetchone()
        assert row == ("My Sweep", "hello")

    def test_parameter_specs_stored(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_parameter("lr", [1e-3, 3e-4], is_static=False)
        exp.add_parameter("env_steps", [1_000_000], is_static=True)
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            rows = con.execute("SELECT name, is_static FROM ParameterSpecs ORDER BY name").fetchall()
        assert ("env_steps", 1) in rows
        assert ("lr", 0) in rows

    def test_metric_specs_stored(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_metric("reward", type="float", frequency="per_episode")
        exp.add_metric("value_loss", type="float", frequency="per_update")
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            rows = {r[0] for r in con.execute("SELECT name FROM MetricSpecs").fetchall()}
        assert rows == {"reward", "value_loss"}

    def test_runs_created_for_seeds(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_parameter("seed", [0, 1, 2])
        exp.add_parameter("lr", [1e-3])
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            count = con.execute("SELECT COUNT(*) FROM Runs").fetchone()[0]
        assert count == 3

    def test_cartesian_product_runs(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_parameter("seed", [0, 1])
        exp.add_parameter("lr", [1e-3, 3e-4])
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            count = con.execute("SELECT COUNT(*) FROM Runs").fetchone()[0]
        # 2 seeds × 2 lr → 4 runs
        assert count == 4

    def test_ablation_doubles_runs(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_parameter("seed", [0])
        exp.add_parameter("use_gae", [True, False])
        exp.add_ablation("no_gae", {"use_gae": False})
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            count = con.execute("SELECT COUNT(*) FROM Runs").fetchone()[0]
        # base: 2 configs × 1 seed = 2; ablation: 2 configs overridden = 2 more
        assert count == 4

    def test_component_version_created(self, db_path: Path) -> None:
        exp = Experiment("Test")
        ppo = Component(name="PPO", path=Path("/nonexistent.py"), type=ComponentType.ALGO)
        with exp.for_component(ppo):
            exp.add_parameter("lr", [1e-3])
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            row = con.execute("SELECT name FROM Components WHERE name='PPO'").fetchone()
        assert row is not None

    def test_hyperparam_config_deduplication(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_parameter("seed", [0])
        exp.add_parameter("lr", [1e-3])
        exp.sync(db_path)
        exp2 = Experiment("Test2")
        exp2.add_parameter("seed", [0])
        exp2.add_parameter("lr", [1e-3])
        exp2.sync(db_path)  # second sync inserts a new experiment but reuses HyperparamConfig
        with sqlite3.connect(db_path) as con:
            count = con.execute("SELECT COUNT(*) FROM HyperparamConfigs").fetchone()[0]
        assert count == 1

    def test_schema_has_all_adr008_tables(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.sync(db_path)
        expected = {
            "Components",
            "ComponentVersions",
            "HyperparamConfigs",
            "Experiments",
            "Runs",
            "Executions",
            "ExecutionRuns",
            "ParameterSpecs",
            "MetricSpecs",
        }
        with sqlite3.connect(db_path) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert expected.issubset(tables)

    def test_conditional_params_only_in_matching_runs(self, db_path: Path) -> None:
        exp = Experiment("Test")
        exp.add_parameter("seed", [0])
        exp.add_parameter("use_gae", [True, False])
        with exp.when(use_gae=True):
            exp.add_parameter("gae_lambda", [0.9, 0.95])
        exp.sync(db_path)
        with sqlite3.connect(db_path) as con:
            blobs = [r[0] for r in con.execute("SELECT json_blob FROM HyperparamConfigs").fetchall()]
        import json

        configs = [json.loads(b) for b in blobs]
        gae_configs = [c for c in configs if c.get("use_gae") is True]
        non_gae_configs = [c for c in configs if c.get("use_gae") is False]
        assert all("gae_lambda" in c for c in gae_configs)
        assert all("gae_lambda" not in c for c in non_gae_configs)
