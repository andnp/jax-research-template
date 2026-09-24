"""Small (unit) tests for experiment-definition — must complete in < 1ms each."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from experiment_definition import Component, ComponentType, Experiment, MetricFrequency, MetricType
from experiment_definition.ablation import AblationSpec
from experiment_definition.db import DatabaseManager
from experiment_definition.metric import MetricSpec
from experiment_definition.parameter import ParameterSpec

# ── Component ─────────────────────────────────────────────────────────────────

class TestComponent:
    def test_defaults(self) -> None:
        c = Component(name="PPO", path=Path("domains/rl/agents/src/rl_agents/ppo.py"))
        assert c.type == ComponentType.OTHER

    def test_explicit_type(self) -> None:
        c = Component(name="PPO", path=Path("p.py"), type=ComponentType.ALGO)
        assert c.type == ComponentType.ALGO

    def test_code_hash_missing_file_returns_empty(self) -> None:
        c = Component(name="X", path=Path("/nonexistent/path.py"))
        assert c.code_hash() == ""

    def test_from_module_resolves_path(self) -> None:
        import experiment_definition.component as component_module

        c = Component.from_module("component", component_module, type=ComponentType.OTHER)
        assert c.path == Path(component_module.__file__ or "")

    def test_from_module_rejects_module_without_source_file(self) -> None:
        import sys

        with pytest.raises(ValueError, match="no source file"):
            Component.from_module("sys", sys)


# ── ParameterSpec ─────────────────────────────────────────────────────────────

class TestParameterSpec:
    def test_basic(self) -> None:
        p = ParameterSpec(name="lr", values=[1e-3, 3e-4])
        assert p.is_static is False
        assert p.conditions == {}

    def test_static_flag(self) -> None:
        p = ParameterSpec(name="env_steps", values=[1_000_000], is_static=True)
        assert p.is_static is True

    def test_with_conditions(self) -> None:
        p = ParameterSpec(name="gae_lambda", values=[0.9, 0.95], conditions={"use_gae": True})
        assert p.conditions == {"use_gae": True}


# ── MetricSpec ────────────────────────────────────────────────────────────────

class TestMetricSpec:
    def test_valid(self) -> None:
        m = MetricSpec(name="reward", type=MetricType.FLOAT, frequency=MetricFrequency.PER_EPISODE)
        assert m.name == "reward"

    def test_enum_values(self) -> None:
        assert MetricFrequency.EVAL_ONLY == "eval_only"
        assert MetricType.FLOAT == "float"


# ── AblationSpec ──────────────────────────────────────────────────────────────

class TestAblationSpec:
    def test_overrides(self) -> None:
        a = AblationSpec(name="no_gae", overrides={"use_gae": False})
        assert a.overrides["use_gae"] is False


# ── Experiment builder ────────────────────────────────────────────────────────

class TestExperimentBuilder:
    def test_add_parameter_global(self) -> None:
        exp = Experiment("Test")
        exp.add_parameter("seed", range(3))
        assert len(exp._state.parameters) == 1
        assert exp._state.parameters[0].name == "seed"
        assert exp._state.parameters[0].values == [0, 1, 2]

    def test_add_parameter_is_static(self) -> None:
        exp = Experiment("Test")
        exp.add_parameter("env_steps", [1_000_000], is_static=True)
        assert exp._state.parameters[0].is_static is True

    def test_for_component_scopes_parameter(self) -> None:
        exp = Experiment("Test")
        ppo = Component(name="PPO", path=Path("p.py"), type=ComponentType.ALGO)
        with exp.for_component(ppo):
            exp.add_parameter("lr", [1e-3])
        p = exp._state.parameters[0]
        assert p.component is not None
        assert p.component.name == "PPO"

    def test_for_component_registers_component(self) -> None:
        exp = Experiment("Test")
        ppo = Component(name="PPO", path=Path("p.py"))
        with exp.for_component(ppo):
            pass
        assert ppo in exp._state.components

    def test_for_component_restores_scope(self) -> None:
        exp = Experiment("Test")
        ppo = Component(name="PPO", path=Path("p.py"))
        with exp.for_component(ppo):
            pass
        assert exp._component_scope is None

    def test_when_attaches_conditions(self) -> None:
        exp = Experiment("Test")
        exp.add_parameter("use_gae", [True, False])
        with exp.when(use_gae=True):
            exp.add_parameter("gae_lambda", [0.9])
        cond_param = exp._state.parameters[1]
        assert cond_param.conditions == {"use_gae": True}

    def test_when_nested_conditions_merge(self) -> None:
        exp = Experiment("Test")
        with exp.when(a=1):
            with exp.when(b=2):
                exp.add_parameter("x", [1])
        assert exp._state.parameters[0].conditions == {"a": 1, "b": 2}

    def test_add_metric(self) -> None:
        exp = Experiment("Test")
        exp.add_metric("reward", kind="float", frequency="per_episode")
        assert len(exp._state.metrics) == 1
        m = exp._state.metrics[0]
        assert m.name == "reward"
        assert m.type == MetricType.FLOAT
        assert m.frequency == MetricFrequency.PER_EPISODE

    def test_add_ablation(self) -> None:
        exp = Experiment("Test")
        exp.add_ablation("no_gae", {"use_gae": False})
        assert len(exp._state.ablations) == 1
        assert exp._state.ablations[0].name == "no_gae"

    def test_fluent_chaining(self) -> None:
        exp = Experiment("Test")
        result = exp.add_parameter("lr", [1e-3]).add_metric("reward", kind="float", frequency="per_episode").add_ablation("x", {})
        assert result is exp

    def test_invalid_metric_type_raises(self) -> None:
        exp = Experiment("Test")
        with pytest.raises(ValueError):
            exp.add_metric("x", kind="invalid", frequency="per_episode")

    def test_invalid_metric_frequency_raises(self) -> None:
        exp = Experiment("Test")
        with pytest.raises(ValueError):
            exp.add_metric("x", kind="float", frequency="invalid")


class TestConditionalSweepAPI:
    def test_resolves_only_valid_conditional_configurations(self) -> None:
        """Expand conditional axes without emitting invalid combinations."""
        from experiment_definition import Sweep, values, when

        sweep = Sweep(
            axes={
                "agent": values("dqn", "ppo"),
                "learning_rate": values(1e-3, 3e-4),
            },
            constraints=[
                when("agent", "dqn").require("learning_rate", 1e-3),
            ],
        )

        assert sweep.resolve() == [
            {"agent": "dqn", "learning_rate": 1e-3},
            {"agent": "ppo", "learning_rate": 1e-3},
            {"agent": "ppo", "learning_rate": 3e-4},
        ]


# ── Component provenance ──────────────────────────────────────────────────────


class TestComponentProvenance:
    def test_single_algo_component_stamps_version(self, tmp_path: Path) -> None:
        """The common case: one ALGO component unambiguously authors every run."""
        db_path = tmp_path / "experiments.sqlite"
        algo = Component(name="PPO", path=Path("/nonexistent/ppo.py"), type=ComponentType.ALGO)
        env = Component(name="Env", path=Path("/nonexistent/env.py"), type=ComponentType.ENV)
        exp = Experiment("SingleAlgo")
        exp.add_parameter("seed", [0])
        with exp.for_component(algo):
            exp.add_parameter("lr", [1e-3])
        with exp.for_component(env):
            exp.add_parameter("gamma", [0.99])
        exp.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("SingleAlgo")
            assert experiment_row is not None
            runs = database.list_runs(experiment_row.id)
            assert len(runs) == 1
            assert runs[0].algo_version_id is not None

    def test_default_other_component_syncs_to_a_fresh_database(self, tmp_path: Path) -> None:
        """A component left at its default OTHER type must be storable, not rejected by the schema."""
        db_path = tmp_path / "experiments.sqlite"
        algo = Component(name="PPO", path=Path("/nonexistent/ppo.py"), type=ComponentType.ALGO)
        model = Component(name="Model", path=Path("/nonexistent/model.py"))
        exp = Experiment("WithOther")
        exp.add_parameter("seed", [0])
        with exp.for_component(algo):
            exp.add_parameter("lr", [1e-3])
        with exp.for_component(model):
            exp.add_parameter("width", [4])
        exp.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("WithOther")
            assert experiment_row is not None
            assert len(database.list_runs(experiment_row.id)) == 1

    def test_two_algo_components_leave_version_null(self, tmp_path: Path) -> None:
        """
        A two-agent bakeoff declares two ALGO components. There is no
        per-run record of which component actually produced a given run, so
        arbitrarily stamping every Run with the first component's version id
        would silently attribute one agent's runs to the other. NULL is the
        honest answer: "unknown", not a specific wrong one.
        """
        db_path = tmp_path / "experiments.sqlite"
        baseline = Component(name="Baseline", path=Path("/nonexistent/baseline.py"), type=ComponentType.ALGO)
        variant = Component(name="Variant", path=Path("/nonexistent/variant.py"), type=ComponentType.ALGO)
        env = Component(name="Env", path=Path("/nonexistent/env.py"), type=ComponentType.ENV)
        exp = Experiment("Bakeoff")
        exp.add_parameter("seed", [0])
        with exp.for_component(baseline):
            pass
        with exp.for_component(variant):
            pass
        with exp.for_component(env):
            exp.add_parameter("gamma", [0.99])
        exp.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("Bakeoff")
            assert experiment_row is not None
            runs = database.list_runs(experiment_row.id)
            assert len(runs) == 1
            assert runs[0].algo_version_id is None


class TestSyncedRunIdentity:
    def test_baseline_sync_deduplicates_nullable_identity(self, tmp_path: Path) -> None:
        """Baseline runs with NULL component and ablation values remain unique."""
        db_path = tmp_path / "experiments.sqlite"
        exp = Experiment("Baseline")
        exp.add_parameter("seed", [0])

        exp.sync(db_path)
        exp.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("Baseline")
            assert experiment_row is not None
            run = database.list_runs(experiment_row.id)[0]
            with pytest.raises(sqlite3.IntegrityError):
                database.conn.execute(
                    "INSERT INTO Runs(experiment_id, algo_version_id, env_version_id, hyper_id, seed, ablation) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (run.experiment_id, run.algo_version_id, run.env_version_id, run.hyper_id, run.seed, None),
                )

    def _experiment(self, name: str, learning_rates: list[float]) -> Experiment:
        algo = Component(name="PPO", path=Path("/nonexistent/ppo.py"), type=ComponentType.ALGO)
        env = Component(name="Env", path=Path("/nonexistent/env.py"), type=ComponentType.ENV)
        exp = Experiment(name)
        exp.add_parameter("seed", [0, 1])
        with exp.for_component(algo):
            exp.add_parameter("lr", learning_rates)
        with exp.for_component(env):
            exp.add_parameter("gamma", [0.99])
        return exp

    def test_sync_returns_every_run_it_wants(self, tmp_path: Path) -> None:
        """The returned ids cover the rows in the table, whether inserted now or already there."""
        db_path = tmp_path / "experiments.sqlite"
        exp = self._experiment("Wanted", [1e-3, 1e-4])

        wanted = exp.sync(db_path)
        wanted_again = exp.sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("Wanted")
            assert experiment_row is not None
            run_ids = {run.id for run in database.list_runs(experiment_row.id)}
        assert wanted == run_ids
        assert wanted_again == wanted

    def test_a_narrowed_definition_leaves_its_dropped_runs_as_orphans(self, tmp_path: Path) -> None:
        """Dropping a swept value does not remove its rows, so they stay plannable forever.

        This is the leak ``prune`` exists to close: the ``Runs`` UNIQUE identity has no
        notion of a row the definition has stopped wanting.
        """
        db_path = tmp_path / "experiments.sqlite"
        self._experiment("Narrowed", [1e-3, 1e-4]).sync(db_path)

        wanted = self._experiment("Narrowed", [1e-3]).sync(db_path)

        with DatabaseManager(db_path) as database:
            database.initialize()
            experiment_row = database.get_experiment("Narrowed")
            assert experiment_row is not None
            assert len(database.list_runs(experiment_row.id)) == 4
            orphans = database.list_orphan_runs(experiment_row.id, wanted)
            assert len(orphans) == 2
            assert database.delete_runs([run.id for run in orphans]) == 2
            assert len(database.list_runs(experiment_row.id)) == 2

    def test_delete_runs_refuses_a_run_an_execution_touched(self, tmp_path: Path) -> None:
        """The guard is inside the delete, so a stale id list cannot remove a result."""
        db_path = tmp_path / "experiments.sqlite"
        wanted = self._experiment("Guarded", [1e-3]).sync(db_path)
        protected = sorted(wanted)[0]

        with DatabaseManager(db_path) as database:
            database.initialize()
            database.link_execution_run(database.add_execution(hostname="somewhere"), protected)

            assert database.delete_runs([protected]) == 0
            experiment_row = database.get_experiment("Guarded")
            assert experiment_row is not None
            assert protected in {run.id for run in database.list_runs(experiment_row.id)}
