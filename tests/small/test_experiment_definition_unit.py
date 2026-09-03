"""Small (unit) tests for experiment-definition — must complete in < 1ms each."""

from __future__ import annotations

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
        c = Component(name="PPO", path=Path("libs/rl-agents/src/rl_agents/ppo.py"))
        assert c.type == ComponentType.OTHER

    def test_explicit_type(self) -> None:
        c = Component(name="PPO", path=Path("p.py"), type=ComponentType.ALGO)
        assert c.type == ComponentType.ALGO

    def test_code_hash_missing_file_returns_empty(self) -> None:
        c = Component(name="X", path=Path("/nonexistent/path.py"))
        assert c.code_hash() == ""


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
